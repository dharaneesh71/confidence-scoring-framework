import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  Container, Paper, TextField, Button, Typography, Box, Alert, CircularProgress
} from '@mui/material';

// ── Same pattern used in QAPage / AdminPage / AuthContext ─────────────────────
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function LoginPage() {
  const [email, setEmail]         = useState('');
  const [password, setPassword]   = useState('');
  const [error, setError]         = useState('');
  const [submitting, setSubmitting] = useState(false);

  // ── NEW: register-mode state ──────────────────────────────────────────────
  const [mode, setMode]       = useState('login');   // 'login' | 'register'
  const [confirm, setConfirm] = useState('');        // confirm-password field

  const { login }  = useAuth();
  const navigate   = useNavigate();

  // ── Existing login handler (unchanged) ───────────────────────────────────
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

  // ── NEW: register handler ─────────────────────────────────────────────────
  const handleRegister = async (e) => {
    if (e) e.preventDefault();
    setError('');

    // Client-side validation
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

      if (res.status === 400) {
        setError('An account with this email already exists.');
        return;
      }
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || 'Registration failed. Please try again.');
        return;
      }

      // Auto-login after successful registration
      const success = await login(email, password);
      if (success) {
        navigate('/');
      } else {
        // Account created but auto-login failed — send them to sign-in
        setMode('login');
        setError('Account created! Please sign in.');
      }
    } catch {
      setError('Network error. Please check your connection.');
    } finally {
      setSubmitting(false);
    }
  };

  // ── NEW: switch modes and reset fields ────────────────────────────────────
  const switchMode = () => {
    setMode(prev => prev === 'login' ? 'register' : 'login');
    setError('');
    setConfirm('');
  };

  // ── Derived: which handler to call on form submit ─────────────────────────
  const handleSubmit = mode === 'login' ? handleLogin : handleRegister;

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

        {/* ── Subtitle changes based on mode ─────────────────────────────── */}
        <Typography variant="body2" color="text.secondary">
          {mode === 'login' ? 'Sign in to access the system' : 'Create a new account'}
        </Typography>

        {error && (
          <Alert severity="error" sx={{ width: '100%' }}>{error}</Alert>
        )}

        <Box component="form" onSubmit={handleSubmit} sx={{ width: '100%', mt: 2 }}>
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

          {/* ── NEW: confirm password — only shown in register mode ─────── */}
          {mode === 'register' && (
            <TextField
              label="Confirm Password"
              type="password"
              fullWidth
              margin="normal"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              disabled={submitting}
            />
          )}

          {/* ── Submit button — label adapts to mode ───────────────────── */}
          <Button
            type="submit"
            fullWidth
            variant="contained"
            onClick={handleSubmit}
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
            {submitting
              ? (mode === 'login' ? 'Signing in...' : 'Creating account...')
              : (mode === 'login' ? 'Sign In' : 'Create Account')
            }
          </Button>
        </Box>

        {/* ── NEW: toggle link between sign-in and register ──────────────── */}
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
          <Box
            component="span"
            onClick={switchMode}
            sx={{
              color: 'primary.main',
              cursor: 'pointer',
              fontWeight: 'bold',
              '&:hover': { textDecoration: 'underline' },
            }}
          >
            {mode === 'login' ? 'Create one' : 'Sign in'}
          </Box>
        </Typography>
      </Paper>
    </Container>
  );
}