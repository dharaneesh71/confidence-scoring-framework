import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  Container, Paper, TextField, Button, Typography, Box, Alert, CircularProgress
} from '@mui/material';

export default function LoginPage() {
  const [email, setEmail]         = useState('');
  const [password, setPassword]   = useState('');
  const [error, setError]         = useState('');
  const [submitting, setSubmitting] = useState(false); // ← new

  const { login }  = useAuth();
  const navigate   = useNavigate();

  // ── Replaced handleSubmit with handleLogin ────────────────────────────────
  const handleLogin = async (e) => {
    if (e) e.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      const success = await login(email, password);
      if (success) {
        navigate('/');
      } else {
        setError('Invalid email or password');
      }
    } finally {
      setSubmitting(false); // always re-enables button, even on network error
    }
  };

  return (
    <Container maxWidth="xs" sx={{ mt: 15 }}>
      <Paper sx={{
        p: 4,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2
      }}>
        <Typography variant="h4" fontWeight="bold" color="primary">
          Confid.AI
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Sign in to access the system
        </Typography>

        {error && (
          <Alert severity="error" sx={{ width: '100%' }}>{error}</Alert>
        )}

        <Box component="form" onSubmit={handleLogin} sx={{ width: '100%', mt: 2 }}>
          <TextField
            label="Email"
            fullWidth
            margin="normal"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={submitting}
          />
          <TextField
            label="Password"
            type="password"
            fullWidth
            margin="normal"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={submitting}
          />

          {/* ── 2. Polished submit button with loading state ───────────── */}
          <Button
            type="submit"
            fullWidth
            variant="contained"
            onClick={handleLogin}
            disabled={submitting}
            startIcon={submitting ? <CircularProgress size={18} color="inherit" /> : null}
            sx={{
              mt: 2,
              py: 1.5,
              borderRadius: 2,
              fontWeight: 'bold',
              textTransform: 'none',
              fontSize: '1rem',
            }}
          >
            {submitting ? 'Signing in...' : 'Sign In'}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
}