import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import {
  AppBar, Toolbar, Typography, Button, Box,
  CssBaseline, ThemeProvider, createTheme, IconButton, Tooltip
} from '@mui/material';
import {
  AdminPanelSettings, Brightness4, Brightness7,
  Logout, ChatBubbleOutline, Menu as MenuIcon
} from '@mui/icons-material';

import QAPage from './pages/QAPage';
import AdminPage from './pages/AdminPage';
import LoginPage from './pages/LoginPage';
import Sidebar from './components/Sidebar';
import { AuthProvider, useAuth } from './context/AuthContext';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const PrivateRoute = ({ children, adminOnly = false }) => {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" />;
  if (adminOnly && user.role !== 'admin') return <Navigate to="/" />;
  return children;
};

function AppContent() {
  const [mode, setMode] = useState('dark');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sessions, setSessions] = useState([]);

  const { user, logout } = useAuth();
  const location = useLocation();

  const theme = React.useMemo(() => createTheme({
    palette: {
      mode,
      primary: { main: mode === 'light' ? '#3F51B5' : '#7986CB' },
      background: {
        default: mode === 'dark' ? '#0f172a' : '#f5f5f5',
        paper: mode === 'dark' ? '#1e293b' : '#ffffff'
      },
    },
  }), [mode]);

  const isLoginPage = location.pathname === '/login';

  useEffect(() => {
    if (user && !isLoginPage) {
      fetch(`${API_BASE}/api/history`, {
        headers: { Authorization: `Bearer ${user.token}` }
      })
        .then(res => res.json())
        .then(data => setSessions(data))
        .catch(err => console.error(err));
    }
  }, [user, isLoginPage, location.pathname]);

  const navColor = mode === 'dark' ? '#ffffff' : '#000000';
  // ✅ CHANGED: dark navbar is transparent so star bg shows through; light stays solid
  const navBg = mode === 'dark' ? 'transparent' : '#ffffff';

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>

        {/* ── SIDEBAR ── */}
        {!isLoginPage && user && (
          <Sidebar
            open={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
            sessions={sessions}
            setSessions={setSessions}
          />
        )}

        {/* ── NAVBAR ── */}
        {!isLoginPage && (
          <AppBar
            position="fixed"
            elevation={0}  // ✅ CHANGED: no shadow — seamless with star bg
            sx={{
              zIndex: (t) => t.zIndex.drawer + 1,
              // ✅ CHANGED: transparent dark / solid frosted light
              bgcolor: `${navBg} !important`,
              backdropFilter: mode === 'dark' ? 'none' : 'blur(12px)',
              borderBottom: `1px solid ${
                mode === 'dark'
                  ? 'rgba(255,255,255,0.05)'   // ✅ CHANGED: barely-visible line, not bright
                  : 'rgba(0,0,0,0.1)'
              } !important`,
              color: `${navColor} !important`,
            }}
          >
            <Toolbar>
              {user && (
                <IconButton
                  onClick={() => setSidebarOpen(true)}
                  sx={{ mr: 2, color: `${navColor} !important` }}
                >
                  <MenuIcon />
                </IconButton>
              )}
              <Typography
                variant="h6"
                component={Link}
                to="/"
                sx={{
                  flexGrow: 1,
                  textDecoration: 'none',
                  color: `${navColor} !important`,
                  fontWeight: 'bold',
                  letterSpacing: 1,
                }}
              >
                CONFID.AI
              </Typography>

              <IconButton
                onClick={() => setMode(prev => prev === 'light' ? 'dark' : 'light')}
                sx={{ color: `${navColor} !important` }}
              >
                {mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
              </IconButton>

              {user ? (
                <>
                  <Tooltip title="New Chat">
                    <Button
                      component={Link}
                      to="/chat"
                      startIcon={<ChatBubbleOutline />}
                      sx={{
                        color: `${navColor} !important`,
                        fontWeight: 'bold',
                        fontSize: '0.85rem',
                        display: { xs: 'none', sm: 'inline-flex' },
                      }}
                    >
                      NEW CHAT
                    </Button>
                  </Tooltip>

                  {user.role === 'admin' && (
                    <Tooltip title="Admin Panel">
                      <Button
                        component={Link}
                        to="/admin"
                        startIcon={<AdminPanelSettings />}
                        sx={{ color: `${navColor} !important`, fontWeight: 'bold', fontSize: '0.85rem' }}
                      >
                        ADMIN
                      </Button>
                    </Tooltip>
                  )}

                  <Tooltip title="Logout">
                    <Button
                      onClick={logout}
                      startIcon={<Logout />}
                      sx={{ color: `${navColor} !important`, fontWeight: 'bold', fontSize: '0.85rem' }}
                    >
                      LOGOUT
                    </Button>
                  </Tooltip>
                </>
              ) : (
                <Button component={Link} to="/login" sx={{ color: `${navColor} !important` }}>
                  LOGIN
                </Button>
              )}
            </Toolbar>
          </AppBar>
        )}

        {/* ── CONTENT AREA ── */}
        <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <Toolbar />
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/" element={<PrivateRoute><Navigate to="/chat" /></PrivateRoute>} />
            <Route path="/chat" element={<PrivateRoute><QAPage clearSelection={() => setSessions(s => [...s])} /></PrivateRoute>} />
            <Route path="/chat/:sessionId" element={<PrivateRoute><QAPage clearSelection={() => setSessions(s => [...s])} /></PrivateRoute>} />
            <Route path="/admin" element={<PrivateRoute adminOnly><AdminPage /></PrivateRoute>} />
          </Routes>
        </Box>

      </Box>
    </ThemeProvider>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <AppContent />
      </Router>
    </AuthProvider>
  );
}
