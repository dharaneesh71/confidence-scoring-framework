import React, { useState, useEffect, useCallback } from 'react';
import {
  Container, Paper, Button, Typography, Box,
  Alert, Card, CardContent, useTheme,
  Grid, Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, IconButton, Chip, Rating
} from '@mui/material';
import { CloudUpload, Delete, Assessment, Description, Forum, Warning } from '@mui/icons-material'; 
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useAuth } from '../context/AuthContext';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';  
const AdminPage = () => {
  const theme = useTheme();
  const { user } = useAuth();
  
  const [stats, setStats] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [feedbacks, setFeedbacks] = useState([]);
  const [lowConfSessions, setLowConfSessions] = useState([]);  
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  const fetchAnalytics = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/analytics`, { headers: { 'Authorization': `Bearer ${user.token}` } });
      if(res.ok) setStats(await res.json());
    } catch (err) { console.error(err); }
  }, [user]);

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/documents`, { headers: { 'Authorization': `Bearer ${user.token}` } });
      if(res.ok) setDocuments(await res.json());
    } catch (err) { console.error(err); }
  }, [user]);

  const fetchFeedbacks = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/feedback`, { headers: { 'Authorization': `Bearer ${user.token}` } });
      if(res.ok) setFeedbacks(await res.json());
    } catch (err) { console.error("Error fetching feedbacks", err); }
  }, [user]);

  const fetchLowConfidence = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/low-confidence`, { headers: { 'Authorization': `Bearer ${user.token}` } });
      if(res.ok) setLowConfSessions(await res.json());
    } catch (err) { console.error("Error fetching low confidence sessions", err); }
  }, [user]);

  useEffect(() => {
    if (user) {
      fetchAnalytics();
      fetchDocuments();
      fetchFeedbacks();
      fetchLowConfidence(); 
    }
  }, [user, fetchAnalytics, fetchDocuments, fetchFeedbacks, fetchLowConfidence]);

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true); setMessage('');
    const formData = new FormData(); formData.append('file', selectedFile);
    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST', headers: { 'Authorization': `Bearer ${user.token}` }, body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setMessage(`✅ Success: ${data.message}`);
        setSelectedFile(null);
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
        method: 'DELETE', headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) fetchDocuments();
    } catch (err) { console.error(err); }
  };

  const cardStyle = { p: 3, borderRadius: '20px', mb: 4, border: `1px solid ${theme.palette.divider}` };

  return (
    <Container maxWidth="lg" sx={{ my: 4, height: "100%", overflowY: "auto" }}>
      <Typography variant="h4" fontWeight={800} gutterBottom sx={{ mb: 4 }}>Admin Dashboard</Typography>
      
      {/* 1. ANALYTICS GRIDS */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ ...cardStyle, height: 320, mb: 0 }}>
             <Typography variant="h6" gutterBottom><Assessment sx={{ verticalAlign: 'middle', mr: 1 }}/> Feedback Ratings</Typography>
             {stats ? (
               <ResponsiveContainer width="100%" height="85%">
                 <BarChart data={stats.distribution}>
                   <XAxis dataKey="name" /> <YAxis /> <Tooltip />
                   <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                     {stats.distribution.map((entry, idx) => (
                       <Cell key={`cell-${idx}`} fill={idx > 2 ? theme.palette.success.main : theme.palette.error.main} />
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
              <Typography variant="h1" fontWeight="bold" color="primary">{stats?.average_rating || 0}</Typography>
              <Chip label={`${stats?.total_feedback || 0} Total Reviews`} sx={{ mt: 3 }} color="primary" variant="outlined" />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper sx={{ ...cardStyle, border: `1px solid ${theme.palette.error.main}` }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
          <Warning sx={{ color: 'error.main' }} />
          <Typography variant="h6" fontWeight="bold" sx={{ color: 'error.main' }}>
            Flagged: Low Confidence Sessions
          </Typography>
          <Chip
            label={`${lowConfSessions.length} sessions need review`}
            color="error" size="small" sx={{ ml: 'auto' }}
          />
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
                  <TableRow
                    key={idx}
                    sx={{
                      bgcolor: theme.palette.mode === 'dark'
                        ? 'rgba(211,47,47,0.08)'
                        : 'rgba(211,47,47,0.04)',
                      '&:hover': {
                        bgcolor: theme.palette.mode === 'dark'
                          ? 'rgba(211,47,47,0.15)'
                          : 'rgba(211,47,47,0.08)',
                      }
                    }}
                  >
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
                      <Chip
                        label={`${Math.round(row.confidence_score * 100)}%`}
                        color="error" size="small" sx={{ fontWeight: 'bold' }}
                      />
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

      {/* 3. USER FEEDBACK TABLE — unchanged */}
      <Paper sx={cardStyle}>
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 3 }}><Forum sx={{ verticalAlign: 'middle', mr: 1 }}/> User Feedback Logs</Typography>
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
                    <TableCell>{fb.user_email || "Unknown User"}</TableCell>
                    <TableCell><Rating value={fb.rating} readOnly size="small" /></TableCell>
                    <TableCell sx={{ color: fb.comment ? 'text.primary' : 'text.secondary', fontStyle: fb.comment ? 'normal' : 'italic' }}>
                      {fb.comment || "No comment provided"}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* 4. DOCUMENT MANAGEMENT — unchanged */}
      <Paper sx={cardStyle}>
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 3 }}><Description sx={{ verticalAlign: 'middle', mr: 1 }}/> Knowledge Base Documents</Typography>
        <Box sx={{ display: 'flex', gap: 2, mb: 4, alignItems: 'center', p: 3, border: `1px dashed ${theme.palette.divider}`, borderRadius: 2 }}>
          <Button variant="contained" component="label" startIcon={<CloudUpload />}>
            Choose PDF <input id="file-input" type="file" hidden accept=".pdf" onChange={(e) => setSelectedFile(e.target.files[0])} />
          </Button>
          <Typography sx={{ flexGrow: 1 }}>{selectedFile ? selectedFile.name : "No file selected"}</Typography>
          <Button variant="contained" color="success" onClick={handleUpload} disabled={!selectedFile || uploading}>
            {uploading ? 'Uploading...' : 'Upload'}
          </Button>
        </Box>
        {message && <Alert severity={message.includes('Success') ? 'success' : 'error'} sx={{ mb: 3 }}>{message}</Alert>}
        
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Filename</TableCell>
                <TableCell align="center">Chunks</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {documents.length === 0 ? (
                <TableRow><TableCell colSpan={3} align="center">No documents found.</TableCell></TableRow>
              ) : (
                documents.map((doc) => (
                  <TableRow key={doc.id}>
                    <TableCell sx={{ fontWeight: 'bold' }}>{doc.filename}</TableCell>
                    <TableCell align="center"><Chip label={doc.chunk_count} size="small" /></TableCell>
                    <TableCell align="right">
                      <IconButton color="error" onClick={() => handleDelete(doc.id, doc.filename)}><Delete /></IconButton>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Container>
  );
};

export default AdminPage;
