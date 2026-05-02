import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  Paper, TextField, Button, Typography, Box, Alert, CircularProgress,
  useTheme  // ✅ CHANGED: added useTheme
} from '@mui/material';
import StarBackground from '../components/Starbackground';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [mode, setMode] = useState('login');
  const [confirm, setConfirm] = useState('');

  const { login } = useAuth();
  const navigate = useNavigate();
  const theme = useTheme(); // ✅ CHANGED: get theme for mode detection

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
      setSubmitting(false);
    }
  };

  const handleRegister = async (e) => {
    if (e) e.preventDefault();
    setError('');
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError('Please enter a valid email address.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    if (password !== confirm) {
      setError('Passwords do not match.');
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch(`${API_BASE}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (res.status === 400) { setError('An account with this email already exists.'); return; }
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || 'Registration failed. Please try again.');
        return;
      }
      const success = await login(email, password);
      if (success) {
        navigate('/');
      } else {
        setMode('login');
        setError('Account created! Please sign in.');
      }
    } catch {
      setError('Network error. Please check your connection.');
    } finally {
      setSubmitting(false);
    }
  };

  const switchMode = () => {
    setMode(prev => prev === 'login' ? 'register' : 'login');
    setError('');
    setConfirm('');
  };

  const handleSubmit = mode === 'login' ? handleLogin : handleRegister;

  return (
    <Box
      sx={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 2,
      }}
    >
      {/* ✅ CHANGED: pass theme mode so stars/bg match dark or light */}
      <StarBackground mode={theme.palette.mode} />

      <Paper
        elevation={6}
        component="form"
        onSubmit={handleSubmit}
        sx={{
          position: 'relative',
          zIndex: 1,
          width: '100%',
          maxWidth: 420,
          p: 4,
          borderRadius: 3,
          bgcolor: theme.palette.mode === 'dark'
            ? 'rgba(10, 14, 26, 0.85)'
            : 'rgba(255,255,255,0.88)',
          backdropFilter: 'blur(12px)',
          border: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'}`,
          boxShadow: '0 8px 40px rgba(0,0,0,0.5)',
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        <Typography variant="h4" fontWeight="bold" textAlign="center" color="primary">
          Confid.AI
        </Typography>
        <Typography variant="body2" textAlign="center" color="text.secondary" mb={1}>
          {mode === 'login' ? 'Sign in to access the system' : 'Create a new account'}
        </Typography>

        {error && <Alert severity="error">{error}</Alert>}

        <TextField
          label="Email"
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          disabled={submitting}
          fullWidth
          required
          autoComplete="email"
        />
        <TextField
          label="Password"
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          disabled={submitting}
          fullWidth
          required
          autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
        />
        {mode === 'register' && (
          <TextField
            label="Confirm Password"
            type="password"
            value={confirm}
            onChange={e => setConfirm(e.target.value)}
            disabled={submitting}
            fullWidth
            required
          />
        )}

        <Button
          type="submit"
          variant="contained"
          fullWidth
          disabled={submitting}
          startIcon={submitting ? <CircularProgress size={18} color="inherit" /> : null}
          sx={{ py: 1.5, borderRadius: 2, fontWeight: 'bold', textTransform: 'none', fontSize: '1rem' }}
        >
          {submitting
            ? (mode === 'login' ? 'Signing in...' : 'Creating account...')
            : (mode === 'login' ? 'Sign In' : 'Create Account')}
        </Button>

        <Typography variant="body2" textAlign="center" color="text.secondary">
          {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
          <Box
            component="span"
            onClick={switchMode}
            sx={{ color: 'primary.main', cursor: 'pointer', fontWeight: 600, '&:hover': { textDecoration: 'underline' } }}
          >
            {mode === 'login' ? 'Create one' : 'Sign in'}
          </Box>
        </Typography>
      </Paper>
    </Box>
  );
}
