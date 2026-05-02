import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Container, Paper, Button, Typography, Box,
  Alert, Card, CardContent, useTheme,
  Grid, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, IconButton, Chip, Rating,
  LinearProgress, TextField
} from '@mui/material';
import {
  CloudUpload, Delete, Assessment, Description,
  Forum, Warning, AutoFixHigh, Category
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useAuth } from '../context/AuthContext';
import StarBackground from '../components/Starbackground';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const AdminPage = () => {
  const theme = useTheme();
  const { user } = useAuth();

  // ── Existing state ─────────────────────────────────────────────────────────
  const [stats, setStats] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [feedbacks, setFeedbacks] = useState([]);
  const [lowConfSessions, setLowConfSessions] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  // ── Retraining state ───────────────────────────────────────────────────────
  const [isRetraining, setIsRetraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingMessage, setTrainingMessage] = useState('');
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const pollingRef = useRef(null);

  // ── Domain state ───────────────────────────────────────────────────────────
  const [domainInput, setDomainInput] = useState('General');

  // ── Task 19 — Model Version History ───────────────────────────────────────
  const [modelHistory, setModelHistory] = useState([]);

  // ── Fetchers ───────────────────────────────────────────────────────────────
  const fetchAnalytics = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/analytics`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) setStats(await res.json());
    } catch (err) { console.error(err); }
  }, [user]);

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/documents`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) setDocuments(await res.json());
    } catch (err) { console.error(err); }
  }, [user]);

  const fetchFeedbacks = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/feedback`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) setFeedbacks(await res.json());
    } catch (err) { console.error('Error fetching feedbacks', err); }
  }, [user]);

  const fetchLowConfidence = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/low-confidence`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) setLowConfSessions(await res.json());
    } catch (err) { console.error('Error fetching low confidence sessions', err); }
  }, [user]);

  const fetchModelHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/model-history`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setModelHistory(data.history || []);
      }
    } catch (err) { console.error('Error fetching model history', err); }
  }, [user]);

  useEffect(() => {
    if (user) {
      fetchAnalytics();
      fetchDocuments();
      fetchFeedbacks();
      fetchLowConfidence();
      fetchModelHistory();
    }
  }, [user, fetchAnalytics, fetchDocuments, fetchFeedbacks, fetchLowConfidence, fetchModelHistory]);

  // ── Retraining polling ─────────────────────────────────────────────────────
  const fetchTrainingStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/training-status`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setTrainingProgress(data.progress ?? 0);
        setTrainingMessage(data.message ?? '');
        if (data.status === 'completed' || data.status === 'failed') {
          setIsRetraining(false);
          setTrainingStatus(data.status);
        }
      }
    } catch (err) { console.error('Error fetching training status', err); }
  }, [user]);

  useEffect(() => {
    if (isRetraining) {
      pollingRef.current = setInterval(fetchTrainingStatus, 2000);
    } else {
      clearInterval(pollingRef.current);
    }
    return () => clearInterval(pollingRef.current);
  }, [isRetraining, fetchTrainingStatus]);

  const triggerRetrain = async () => {
    setIsRetraining(true);
    setTrainingProgress(0);
    setTrainingMessage('Initiating retraining pipeline...');
    setTrainingStatus('running');
    try {
      const res = await fetch(`${API_BASE}/api/admin/trigger-retrain`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Failed to start retraining');
      }
    } catch (err) {
      setIsRetraining(false);
      setTrainingStatus('failed');
      setTrainingMessage(`❌ ${err.message}`);
    }
  };

  // ── Upload handler — now includes domain ──────────────────────────────────
  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true); setMessage('');
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('domain', domainInput.trim() || 'General');  // ← domain
    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user.token}` },
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setMessage(`✅ Success: ${data.message}`);
        setSelectedFile(null);
        setDomainInput('General');
        document.getElementById('file-input').value = '';
        fetchDocuments(); fetchAnalytics();
      } else setMessage(`❌ Error: ${data.detail}`);
    } catch (err) { setMessage('❌ Upload failed'); }
    finally { setUploading(false); }
  };

  const handleDelete = async (id, filename) => {
    if (!window.confirm(`Delete "${filename}"?`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/admin/documents/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) fetchDocuments();
    } catch (err) { console.error(err); }
  };

  const cardStyle = {
    p: 3, borderRadius: '20px', mb: 4,
    background: theme.palette.mode === 'dark' ? '#161b27' : '#ffffff',  // ✅ solid card bg
    border: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.06)'}`,
    boxShadow: theme.palette.mode === 'dark'
      ? '0 2px 16px rgba(0,0,0,0.6)'
      : '0 8px 32px rgba(0,0,0,0.08)',
  };

  const statusChipColor = { idle: 'default', running: 'warning', completed: 'success', failed: 'error' }[trainingStatus];
  const statusChipLabel = { idle: 'Idle', running: 'Training...', completed: 'Completed', failed: 'Failed' }[trainingStatus];

  // Domain chip colors — cycle through palette
  const domainColors = ['primary', 'secondary', 'success', 'warning', 'info'];
  const getDomainColor = (domain) => {
    const domains = [...new Set(documents.map(d => d.domain || 'General'))];
    return domainColors[domains.indexOf(domain) % domainColors.length];
  };
  
  return (
    <Box
      sx={{
        position: 'fixed',
        inset: 0,
        top: '64px',
        bgcolor: '#000000',           
        overflowY: 'auto',
        overflowX: 'hidden',
      }}
    >
      <StarBackground mode={theme.palette.mode} /> 
      <Container
        maxWidth="lg"
        sx={{ py: 4, position: 'relative', zIndex: 1 }}
      >
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Admin Dashboard
        </Typography>

      {/* 1. ANALYTICS GRIDS */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ ...cardStyle, height: 320, mb: 0 }}>
            <Typography variant="h6" gutterBottom>
              <Assessment sx={{ verticalAlign: 'middle', mr: 1 }} /> Feedback Ratings
            </Typography>
            {stats ? (
              <ResponsiveContainer width="100%" height="85%">
                <BarChart data={stats.distribution}>
                  <XAxis dataKey="name" /> <YAxis /> <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {stats.distribution.map((entry, idx) => (
                      <Cell key={`cell-${idx}`}
                        fill={idx > 2 ? theme.palette.success.main : theme.palette.error.main}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <Typography>Loading stats...</Typography>}
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ ...cardStyle, height: 320, mb: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center', textAlign: 'center' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>AVERAGE RATING</Typography>
              <Typography variant="h1" fontWeight="bold" color="primary">
                {stats?.average_rating || 0}
              </Typography>
              <Chip label={`${stats?.total_feedback || 0} Total Reviews`} sx={{ mt: 3 }} color="primary" variant="outlined" />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* 2. LOW CONFIDENCE FLAGS */}
      <Paper sx={{ ...cardStyle, border: `1px solid ${theme.palette.error.main}` }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
          <Warning sx={{ color: 'error.main' }} />
          <Typography variant="h6" fontWeight="bold" sx={{ color: 'error.main' }}>
            Flagged: Low Confidence Sessions
          </Typography>
          <Chip label={`${lowConfSessions.length} sessions need review`} color="error" size="small" sx={{ ml: 'auto' }} />
        </Box>
        <TableContainer sx={{ maxHeight: 300 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell>User</TableCell>
                <TableCell>Question</TableCell>
                <TableCell>Answer (preview)</TableCell>
                <TableCell align="center">Score</TableCell>
                <TableCell>Date</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {lowConfSessions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} align="center" sx={{ color: 'success.main', py: 3 }}>
                    ✅ No low confidence sessions found!
                  </TableCell>
                </TableRow>
              ) : (
                lowConfSessions.map((row, idx) => (
                  <TableRow key={idx} sx={{
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(211,47,47,0.08)' : 'rgba(211,47,47,0.04)',
                    '&:hover': { bgcolor: theme.palette.mode === 'dark' ? 'rgba(211,47,47,0.15)' : 'rgba(211,47,47,0.08)' }
                  }}>
                    <TableCell sx={{ fontSize: '0.75rem' }}>{row.user_email}</TableCell>
                    <TableCell sx={{ maxWidth: 200 }}>
                      <Typography variant="body2" noWrap title={row.question}>{row.question}</Typography>
                    </TableCell>
                    <TableCell sx={{ maxWidth: 250 }}>
                      <Typography variant="body2" noWrap sx={{ color: 'text.secondary', fontStyle: 'italic' }} title={row.answer}>
                        {row.answer}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Chip label={`${Math.round(row.confidence_score * 100)}%`} color="error" size="small" sx={{ fontWeight: 'bold' }} />
                    </TableCell>
                    <TableCell sx={{ fontSize: '0.75rem' }}>
                      {new Date(row.timestamp).toLocaleDateString()}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
      
      {/* 3. MODEL RETRAINING PANEL */}
      <Paper sx={{ ...cardStyle, border: `1px solid ${theme.palette.primary.main}` }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
          <AutoFixHigh sx={{ color: 'primary.main' }} />
          <Typography variant="h6" fontWeight="bold" sx={{ color: 'primary.main' }}>Model Retraining</Typography>
          <Chip label={statusChipLabel} color={statusChipColor} size="small" sx={{ ml: 'auto', fontWeight: 'bold' }} />
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Button
            variant="contained" color="primary" startIcon={<AutoFixHigh />}
            onClick={triggerRetrain} disabled={isRetraining}
            sx={{ alignSelf: 'flex-start', borderRadius: 2, fontWeight: 'bold', px: 3 }}
          >
            {isRetraining ? 'Training in Progress...' : 'Trigger Retraining'}
          </Button>
          {(isRetraining || trainingStatus === 'completed' || trainingStatus === 'failed') && (
            <Box sx={{ mt: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="body2" color="text.secondary">{trainingMessage}</Typography>
                <Typography variant="body2" fontWeight="bold" color="primary">{trainingProgress}%</Typography>
              </Box>
              <LinearProgress
                variant="determinate" value={trainingProgress}
                color={trainingStatus === 'failed' ? 'error' : trainingStatus === 'completed' ? 'success' : 'primary'}
                sx={{ height: 10, borderRadius: 5 }}
              />
            </Box>
          )}
          {trainingStatus === 'completed' && (
            <Alert severity="success" sx={{ borderRadius: 2 }}>Retraining completed successfully! The model has been updated.</Alert>
          )}
          {trainingStatus === 'failed' && (
            <Alert severity="error" sx={{ borderRadius: 2 }}>Retraining failed: {trainingMessage || 'Check server logs.'}</Alert>
          )}
        </Box>
      </Paper>

      {/* TASK 19 — MODEL VERSION HISTORY */}
      <Paper sx={{ ...cardStyle, border: `1px solid ${theme.palette.success.main}` }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
          <Assessment sx={{ color: 'success.main' }} />
          <Typography variant="h6" fontWeight="bold" sx={{ color: 'success.main' }}>
            Model Version History
          </Typography>
          <Chip label={`${modelHistory.length} versions`} color="success" size="small" sx={{ ml: 'auto' }} />
        </Box>
        {modelHistory.length === 0 ? (
          <Typography color="text.secondary" sx={{ textAlign: 'center', py: 3 }}>
            No model versions yet. Trigger retraining to create the first version.
          </Typography>
        ) : (
          <>
            {/* Bar chart — accuracy and F1 across versions */}
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={modelHistory} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <XAxis dataKey="version" />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}
                     fill={theme.palette.success.main} />
                <Bar dataKey="f1_score" name="F1 Score" radius={[4, 4, 0, 0]}
                     fill={theme.palette.primary.main} />
              </BarChart>
            </ResponsiveContainer>

            {/* Version table */}
            <TableContainer sx={{ mt: 2 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Version</TableCell>
                    <TableCell align="center">Accuracy</TableCell>
                    <TableCell align="center">F1 Score</TableCell>
                    <TableCell align="center">Loss</TableCell>
                    <TableCell>Trained At</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {modelHistory.map((v, idx) => (
                    <TableRow key={idx} hover>
                      <TableCell>
                        <Chip label={v.version} size="small" color="success" variant="outlined" />
                      </TableCell>
                      <TableCell align="center">{(v.accuracy * 100).toFixed(1)}%</TableCell>
                      <TableCell align="center">{(v.f1_score * 100).toFixed(1)}%</TableCell>
                      <TableCell align="center">{v.loss.toFixed(4)}</TableCell>
                      <TableCell sx={{ fontSize: '0.75rem' }}>
                        {new Date(v.timestamp).toLocaleString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        )}
      </Paper>

      {/* 4. USER FEEDBACK TABLE */}
      <Paper sx={cardStyle}>
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 3 }}>
          <Forum sx={{ verticalAlign: 'middle', mr: 1 }} /> User Feedback Logs
        </Typography>
        <TableContainer sx={{ maxHeight: 300 }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Date</TableCell>
                <TableCell>User</TableCell>
                <TableCell>Rating</TableCell>
                <TableCell>Comment</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {feedbacks.length === 0 ? (
                <TableRow><TableCell colSpan={4} align="center">No feedback recorded yet.</TableCell></TableRow>
              ) : (
                feedbacks.map((fb, idx) => (
                  <TableRow key={idx} hover>
                    <TableCell>{new Date(fb.timestamp).toLocaleDateString()}</TableCell>
                    <TableCell>{fb.user_email || 'Unknown User'}</TableCell>
                    <TableCell><Rating value={fb.rating} readOnly size="small" /></TableCell>
                    <TableCell sx={{ color: fb.comment ? 'text.primary' : 'text.secondary', fontStyle: fb.comment ? 'normal' : 'italic' }}>
                      {fb.comment || 'No comment provided'}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* 5. DOCUMENT MANAGEMENT — WITH DOMAIN */}
      <Paper sx={cardStyle}>
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 3 }}>
          <Description sx={{ verticalAlign: 'middle', mr: 1 }} /> Knowledge Base Documents
        </Typography>

        {/* Upload box — now with domain input */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mb: 4, p: 3,
                   border: `1px dashed ${theme.palette.divider}`, borderRadius: 2 }}>

          {/* Row 1: File picker */}
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <Button variant="contained" component="label" startIcon={<CloudUpload />}>
              Choose PDF
              <input id="file-input" type="file" hidden accept=".pdf"
                onChange={(e) => setSelectedFile(e.target.files[0])} />
            </Button>
            <Typography sx={{ flexGrow: 1, color: selectedFile ? 'text.primary' : 'text.secondary' }}>
              {selectedFile ? selectedFile.name : 'No file selected'}
            </Typography>
          </Box>

          {/* Row 2: Domain + Upload */}
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <Category sx={{ color: 'text.secondary' }} />
            <TextField
              label="Domain / Topic"
              placeholder="e.g. Medical, Legal, Finance"
              value={domainInput}
              onChange={(e) => setDomainInput(e.target.value)}
              size="small"
              sx={{ width: 280 }}
              helperText="Users will filter questions by this domain"
            />
            <Button
              variant="contained" color="success" onClick={handleUpload}
              disabled={!selectedFile || uploading}
              sx={{ height: 40, px: 3, fontWeight: 'bold' }}
            >
              {uploading ? 'Uploading...' : 'Upload'}
            </Button>
          </Box>
        </Box>

        {message && (
          <Alert severity={message.includes('Success') ? 'success' : 'error'} sx={{ mb: 3 }}>
            {message}
          </Alert>
        )}

        {/* Documents table — now with Domain column */}
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Filename</TableCell>
                <TableCell align="center">Domain</TableCell>
                <TableCell align="center">Chunks</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {documents.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} align="center">No documents found.</TableCell>
                </TableRow>
              ) : (
                documents.map((doc) => (
                  <TableRow key={doc.id} hover>
                    <TableCell sx={{ fontWeight: 'bold' }}>{doc.filename}</TableCell>
                    <TableCell align="center">
                      <Chip
                        label={doc.domain || 'General'}
                        size="small"
                        color={getDomainColor(doc.domain || 'General')}
                        variant="outlined"
                        sx={{ fontWeight: 'bold' }}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip label={doc.chunk_count} size="small" />
                    </TableCell>
                    <TableCell align="right">
                      <IconButton color="error" onClick={() => handleDelete(doc.id, doc.filename)}>
                        <Delete />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Container>
    </Box>
  );
};

export default AdminPage;