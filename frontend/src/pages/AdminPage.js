/**
 * Admin Upload Page for adding documents to knowledge base
 */
import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import { CloudUpload, CheckCircle, Login } from '@mui/icons-material';
import { uploadDocument, getStatus } from '../services/api';
import '../styles/AdminPage.css';

const AdminPage = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);

  // Handle login
  const handleLogin = (e) => {
    e.preventDefault();
    
    // Simple validation (in production, validate against backend)
    if (username === 'admin' && password === 'admin123') {
      setIsLoggedIn(true);
      setLoginError('');
      loadSystemStatus();
    } else {
      setLoginError('Invalid username or password');
    }
  };

  // Load system status
  const loadSystemStatus = async () => {
    try {
      const status = await getStatus();
      setSystemStatus(status);
    } catch (err) {
      console.error('Failed to load status:', err);
    }
  };

  // Handle file selection
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type !== 'application/pdf') {
        setUploadError('Only PDF files are allowed');
        setSelectedFile(null);
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setUploadError('File size must be less than 10MB');
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setUploadError(null);
      setUploadResult(null);
    }
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadError(null);
    setUploadResult(null);

    try {
      const result = await uploadDocument(selectedFile, username, password);
      setUploadResult(result);
      setSelectedFile(null);
      // Reset file input
      document.getElementById('file-input').value = '';
      // Reload status
      loadSystemStatus();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  // Login Page
  if (!isLoggedIn) {
    return (
      <Container maxWidth="sm" className="admin-container">
        <Box sx={{ my: 8 }}>
          <Paper elevation={3} sx={{ p: 4 }}>
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <Login sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h4" component="h1" gutterBottom>
                Admin Login
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Sign in to upload documents to the knowledge base
              </Typography>
            </Box>

            <form onSubmit={handleLogin}>
              <TextField
                fullWidth
                label="Username"
                variant="outlined"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                sx={{ mb: 2 }}
                required
              />
              <TextField
                fullWidth
                type="password"
                label="Password"
                variant="outlined"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                sx={{ mb: 3 }}
                required
              />
              {loginError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {loginError}
                </Alert>
              )}
              <Button
                type="submit"
                variant="contained"
                size="large"
                fullWidth
                startIcon={<Login />}
              >
                Login
              </Button>
            </form>

            <Alert severity="info" sx={{ mt: 3 }}>
              <strong>Demo Credentials:</strong><br />
              Username: admin<br />
              Password: admin123
            </Alert>
          </Paper>
        </Box>
      </Container>
    );
  }

  // Main Admin Page
  return (
    <Container maxWidth="lg" className="admin-container">
      <Box sx={{ my: 4 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <div>
            <Typography variant="h3" component="h1" gutterBottom>
              Admin Panel
            </Typography>
            <Typography variant="subtitle1" color="text.secondary">
              Upload PDF documents to the knowledge base
            </Typography>
          </div>
          <Button variant="outlined" onClick={() => setIsLoggedIn(false)}>
            Logout
          </Button>
        </Box>

        {/* System Status */}
        {systemStatus && (
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Overall Status"
                    secondary={systemStatus.status.toUpperCase()}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Documents in Knowledge Base"
                    secondary={systemStatus.documents_count}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Knowledge Base Ready"
                    secondary={systemStatus.knowledge_base_ready ? 'Yes' : 'No'}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="LLM Ready"
                    secondary={systemStatus.llm_ready ? 'Yes' : 'No'}
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        )}

        {/* Upload Form */}
        <Paper elevation={3} sx={{ p: 4 }}>
          <Typography variant="h5" gutterBottom>
            Upload Document
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Upload a PDF document to add it to the knowledge base. The document will be processed,
            chunked, and made available for semantic search.
          </Typography>

          <Divider sx={{ my: 3 }} />

          {/* File Input */}
          <Box sx={{ mb: 3 }}>
            <input
              id="file-input"
              type="file"
              accept=".pdf"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
            <label htmlFor="file-input">
              <Button
                variant="outlined"
                component="span"
                startIcon={<CloudUpload />}
                size="large"
                fullWidth
              >
                Select PDF File
              </Button>
            </label>
          </Box>

          {/* Selected File Display */}
          {selectedFile && (
            <Alert severity="info" sx={{ mb: 3 }}>
              <strong>Selected:</strong> {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
            </Alert>
          )}

          {/* Upload Button */}
          <Button
            variant="contained"
            size="large"
            fullWidth
            disabled={!selectedFile || uploading}
            onClick={handleUpload}
            startIcon={uploading ? <CircularProgress size={20} /> : <CloudUpload />}
          >
            {uploading ? 'Uploading and Processing...' : 'Upload Document'}
          </Button>
        </Paper>

        {/* Error Display */}
        {uploadError && (
          <Alert severity="error" sx={{ mt: 3 }} onClose={() => setUploadError(null)}>
            {uploadError}
          </Alert>
        )}

        {/* Success Display */}
        {uploadResult && (
          <Alert
            severity="success"
            icon={<CheckCircle />}
            sx={{ mt: 3 }}
            onClose={() => setUploadResult(null)}
          >
            <Typography variant="subtitle2" gutterBottom>
              <strong>{uploadResult.message}</strong>
            </Typography>
            <Typography variant="body2">
              File: {uploadResult.filename}<br />
              Chunks created: {uploadResult.chunks_created}
            </Typography>
          </Alert>
        )}

        {/* Instructions */}
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Instructions
            </Typography>
            <List>
              <ListItem>
                <ListItemText
                  primary="1. Select a PDF File"
                  secondary="Click 'Select PDF File' and choose a document from your computer"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="2. Verify File Selection"
                  secondary="Check that the correct file is selected and size is under 10MB"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="3. Upload Document"
                  secondary="Click 'Upload Document' to process and add to knowledge base"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="4. Wait for Processing"
                  secondary="The system will extract text, create chunks, and generate embeddings"
                />
              </ListItem>
            </List>
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
};

export default AdminPage;