import React, { useState } from 'react';
import {
  Container, Paper, TextField, Button, Typography, Box,
  Alert, Card, CardContent, List, ListItem, ListItemText, Chip, useTheme
} from '@mui/material';
import { CloudUpload, Login } from '@mui/icons-material';
import { uploadDocument, getStatus } from '../services/api';
import '../styles/AdminPage.css';

const AdminPage = () => {
  const theme = useTheme();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState({}); 
  const [uploadError, setUploadError] = useState(null);
  const [systemStatus, setSystemStatus] = useState({}); 

  const handleLogin = (e) => {
    e.preventDefault();
    if (username === 'admin' && password === 'admin123') {
      setIsLoggedIn(true); setLoginError(''); loadSystemStatus();
    } else { setLoginError('Invalid username or password'); }
  };

  const loadSystemStatus = async () => {
    try { const status = await getStatus(); setSystemStatus(status); } 
    catch (err) { console.error(err); }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) { setSelectedFile(file); setUploadError(null); setUploadResult(null); }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true); setUploadError(null); setUploadResult(null);
    try {
      const result = await uploadDocument(selectedFile, username, password);
      setUploadResult(result); setSelectedFile(null); document.getElementById('file-input').value = ''; loadSystemStatus();
    } catch (err) { setUploadError(err.message); } finally { setUploading(false); }
  };

  // Dynamic Card Style
  const cardStyle = {
    p: 4,
    borderRadius: '20px',
    border: `1px solid ${theme.palette.divider}`,
    background: theme.palette.mode === 'dark' ? 'rgba(15, 23, 42, 0.8)' : 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    boxShadow: theme.palette.mode === 'light' ? '0 4px 20px rgba(0,0,0,0.1)' : undefined
  };

  // Common Input Style for Glow Effect
  const glowInputStyle = {
    mb: 3,
    '& .MuiOutlinedInput-root': {
      backgroundColor: 'rgba(0,0,0,0.2)', // Slight dark tint inside
      transition: 'all 0.3s ease-in-out',
      // HOVER STATE (The "Light Up" Effect)
      '&:hover fieldset': {
        borderColor: 'secondary.main',
        boxShadow: '0 0 15px rgba(77, 208, 225, 0.6)', // Glow on hover
      },
      // FOCUS STATE (Stronger Glow)
      '&.Mui-focused fieldset': {
        borderColor: 'secondary.main',
        boxShadow: '0 0 25px rgba(77, 208, 225, 0.8)', // Intense glow on focus
      },
    },
    // Label Styling
    '& .MuiInputLabel-root': {
      color: 'text.secondary',
      '&.Mui-focused': { color: 'secondary.main' }
    }
  };

  // LOGIN SCREEN
  if (!isLoggedIn) {
    return (
      <Container maxWidth="xs">
        <Box sx={{ my: 5, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <Paper elevation={12} sx={cardStyle}>
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <Login sx={{ fontSize: 50, color: 'secondary.main', mb: 1 }} />
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary' }}>
                Admin Portal
              </Typography>
            </Box>

            <form onSubmit={handleLogin}>
              <TextField
                fullWidth
                label="Username"
                variant="outlined"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                sx={glowInputStyle} // Applying the glow style here
              />
              <TextField
                fullWidth
                type="password"
                label="Password"
                variant="outlined"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                sx={glowInputStyle} // Applying the glow style here
              />
              {loginError && <Alert severity="error" sx={{ mb: 2 }}>{loginError}</Alert>}
              <Button 
                type="submit" 
                variant="contained" 
                color="secondary" 
                fullWidth 
                size="large" 
                sx={{ 
                  fontWeight: 'bold', 
                  color: 'white',
                  boxShadow: '0 0 10px rgba(77, 208, 225, 0.4)', // Subtle glow for button too
                  '&:hover': { boxShadow: '0 0 20px rgba(77, 208, 225, 0.7)' } 
                }}
              >
                Sign In
              </Button>
            </form>
            <Alert severity="info" sx={{ mt: 3, py: 0, bgcolor: 'rgba(77, 208, 225, 0.1)', color: 'text.primary' }}>
              <small>User: admin | Pass: admin123</small>
            </Alert>
          </Paper>
        </Box>
      </Container>
    );
  }

  // MAIN ADMIN SCREEN
  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Typography variant="h4" sx={{ fontWeight: 800, color: 'text.primary' }}>Knowledge Base</Typography>
          <Button variant="outlined" color="secondary" onClick={() => setIsLoggedIn(false)}>Logout</Button>
        </Box>

        {/* Status Card */}
        {systemStatus && systemStatus.status && (
          <Card sx={{ ...cardStyle, mb: 4, borderLeft: `4px solid ${theme.palette.secondary.main}`, p: 0 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ color: 'secondary.main' }}>System Status</Typography>
              <Box sx={{ display: 'flex', gap: 4 }}>
                 <Box><Typography variant="caption" color="text.secondary">STATUS</Typography><Typography variant="body1" sx={{ fontWeight: 'bold', color: 'success.main' }}>{systemStatus.status.toUpperCase()}</Typography></Box>
                 <Box><Typography variant="caption" color="text.secondary">TOTAL CHUNKS</Typography><Typography variant="body1" sx={{ fontWeight: 'bold', color: 'text.primary' }}>{systemStatus.documents_count}</Typography></Box>
              </Box>
            </CardContent>
          </Card>
        )}

        {/* Upload Card */}
        <Paper sx={cardStyle}>
          <Box sx={{ border: `2px dashed ${theme.palette.divider}`, borderRadius: '12px', p: 4, textAlign: 'center' }}>
            <input id="file-input" type="file" accept=".pdf" onChange={handleFileSelect} style={{ display: 'none' }} />
            <label htmlFor="file-input">
              <Button variant="contained" component="span" startIcon={<CloudUpload />} size="large" sx={{ mb: 2, color: 'white' }}>Select PDF Document</Button>
            </label>
            {selectedFile && <Typography sx={{ mt: 2, color: 'secondary.main' }}>Selected: {selectedFile.name}</Typography>}
          </Box>

          <Button variant="contained" color="primary" fullWidth size="large" disabled={!selectedFile || uploading} onClick={handleUpload} sx={{ mt: 3, color: 'white' }}>
            {uploading ? 'Processing AI Vectorization...' : 'Upload & Process'}
          </Button>
        </Paper>

        {uploadResult && uploadResult.message && <Alert severity="success" sx={{ mt: 3 }}>{uploadResult.message}</Alert>}
      </Box>
    </Container>
  );
};

export default AdminPage;