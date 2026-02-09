import React, { useState, useEffect } from 'react';
import {
  Container, Paper, Button, Typography, Box,
  Alert, Card, CardContent, useTheme, LinearProgress,
  Grid, Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, IconButton, Chip
} from '@mui/material';
import { CloudUpload, Delete, Assessment, Description } from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useAuth } from '../context/AuthContext';
import '../styles/AdminPage.css';

const AdminPage = () => {
  const theme = useTheme();
  const { user } = useAuth();
  
  // State
  const [stats, setStats] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  // Initial Load
  useEffect(() => {
    if (user) {
      fetchAnalytics();
      fetchDocuments();
    }
  }, [user]);

  // --- API CALLS ---

  const fetchAnalytics = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/admin/analytics', {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      const data = await res.json();
      setStats(data);
    } catch (err) { console.error("Error fetching analytics:", err); }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/admin/documents', {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      const data = await res.json();
      setDocuments(data);
    } catch (err) { console.error("Error fetching documents:", err); }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setMessage('');
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user.token}` },
        body: formData,
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setMessage(`✅ Success: ${data.message}`);
        setSelectedFile(null);
        // Reset file input
        const fileInput = document.getElementById('file-input');
        if(fileInput) fileInput.value = '';
        
        fetchDocuments(); // Refresh list
        fetchAnalytics(); // Refresh stats (if doc count changed)
      } else {
        setMessage(`❌ Error: ${data.detail}`);
      }
    } catch (err) { 
        setMessage('❌ Upload failed: Server error'); 
    } finally { 
        setUploading(false); 
    }
  };

  const handleDelete = async (id, filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"? This will remove all knowledge associated with this file.`)) return;
    try {
      const res = await fetch(`http://localhost:8000/api/admin/documents/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      
      if (res.ok) {
          fetchDocuments(); // Refresh list
      } else {
          alert("Failed to delete document");
      }
    } catch (err) { console.error(err); }
  };

  // --- STYLES ---
  const cardStyle = {
    p: 3,
    borderRadius: '20px',
    border: `1px solid ${theme.palette.divider}`,
    background: theme.palette.mode === 'dark' ? 'rgba(15, 23, 42, 0.8)' : 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    boxShadow: theme.palette.mode === 'light' ? '0 4px 20px rgba(0,0,0,0.1)' : undefined
  };

  return (
    <Container maxWidth="lg" sx={{ my: 4 }}>
      <Typography variant="h4" fontWeight={800} gutterBottom sx={{ mb: 4 }}>
        Admin Dashboard
      </Typography>
      
      {/* 1. ANALYTICS SECTION */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Rating Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ ...cardStyle, height: 320 }}>
             <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
               <Assessment color="primary"/> User Feedback Ratings
             </Typography>
             {stats ? (
               <ResponsiveContainer width="100%" height="85%">
                 <BarChart data={stats.distribution}>
                   <XAxis dataKey="name" stroke={theme.palette.text.secondary} />
                   <YAxis stroke={theme.palette.text.secondary} />
                   <Tooltip 
                        contentStyle={{ 
                            backgroundColor: theme.palette.background.paper,
                            borderRadius: '8px',
                            border: `1px solid ${theme.palette.divider}`
                        }} 
                        cursor={{fill: 'transparent'}} 
                   />
                   <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                     {stats.distribution.map((entry, index) => (
                       <Cell key={`cell-${index}`} fill={index > 2 ? theme.palette.success.main : theme.palette.warning.main} />
                     ))}
                   </Bar>
                 </BarChart>
               </ResponsiveContainer>
             ) : <Typography>Loading stats...</Typography>}
          </Paper>
        </Grid>
        
        {/* Summary Card */}
        <Grid item xs={12} md={4}>
          <Card sx={{ ...cardStyle, height: 320, display: 'flex', flexDirection: 'column', justifyContent: 'center', textAlign: 'center' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>AVERAGE RATING</Typography>
              <Typography variant="h1" fontWeight="bold" color="primary">
                {stats?.average_rating || 0}
              </Typography>
              <Typography variant="h5" color="text.secondary" sx={{ opacity: 0.7 }}>/ 5.0</Typography>
              <Chip 
                label={`${stats?.total_feedback || 0} Total Reviews`} 
                sx={{ mt: 3, fontWeight: 'bold' }} 
                color="primary" 
                variant="outlined" 
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* 2. DOCUMENT MANAGEMENT SECTION */}
      <Paper sx={{ ...cardStyle, mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6" fontWeight="bold" sx={{ display: 'flex', alignItems: 'center' }}>
            <Description sx={{ mr: 1 }} />
            Knowledge Base Documents
          </Typography>
        </Box>

        {/* Upload Area */}
        <Box sx={{ 
            display: 'flex', gap: 2, mb: 4, alignItems: 'center', 
            bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)', 
            p: 3, borderRadius: '16px', border: `1px dashed ${theme.palette.divider}`
        }}>
          <Button variant="contained" component="label" startIcon={<CloudUpload />} size="medium">
            Choose PDF
            <input id="file-input" type="file" hidden accept=".pdf" onChange={(e) => setSelectedFile(e.target.files[0])} />
          </Button>
          <Typography sx={{ flexGrow: 1, color: 'text.secondary' }}>
            {selectedFile ? selectedFile.name : "No file selected"}
          </Typography>
          <Button 
            variant="contained" 
            color="success" 
            onClick={handleUpload} 
            disabled={!selectedFile || uploading}
          >
            {uploading ? 'Uploading...' : 'Upload'}
          </Button>
        </Box>
        
        {message && <Alert severity={message.includes('Success') ? 'success' : 'error'} sx={{ mb: 3 }}>{message}</Alert>}

        {/* Document List Table */}
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Filename</TableCell>
                <TableCell align="center">Chunks</TableCell>
                <TableCell align="center">Upload Date</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {documents.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} align="center" sx={{ py: 3, color: 'text.secondary' }}>
                    No documents found in the Knowledge Base.
                  </TableCell>
                </TableRow>
              ) : (
                documents.map((doc) => (
                  <TableRow key={doc.id} hover>
                    <TableCell sx={{ fontWeight: 'bold' }}>{doc.filename}</TableCell>
                    <TableCell align="center">
                        <Chip label={doc.chunk_count} size="small" />
                    </TableCell>
                    <TableCell align="center">{new Date(doc.upload_date).toLocaleDateString()}</TableCell>
                    <TableCell align="right">
                      <IconButton color="error" onClick={() => handleDelete(doc.id, doc.filename)} size="small">
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
  );
};

export default AdminPage;