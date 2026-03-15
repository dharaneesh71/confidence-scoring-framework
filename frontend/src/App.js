import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import {
  AppBar, Toolbar, Typography, Button, Box,
  CssBaseline, ThemeProvider, createTheme, IconButton, Drawer, List, ListItem, ListItemText, Divider, ListItemButton
} from '@mui/material';
import { 
  AdminPanelSettings, Brightness4, Brightness7, 
  Logout, History, Menu as MenuIcon, Settings, ChatBubbleOutline 
} from '@mui/icons-material';

import QAPage from './pages/QAPage';
import AdminPage from './pages/AdminPage';
import LoginPage from './pages/LoginPage';
import { AuthProvider, useAuth } from './context/AuthContext';
import './App.css';

// --- SIDEBAR COMPONENT ---
const Sidebar = ({ open, onClose, onSelectSession }) => {
  const [sessions, setSessions] = useState([]);
  const { user } = useAuth();

  useEffect(() => {
    if (open && user) {
      fetch('http://localhost:8000/api/history', {
        headers: { Authorization: `Bearer ${user.token}` }
      })
      .then(res => res.json())
      .then(data => setSessions(data))
      .catch(err => console.error(err));
    }
  }, [open, user]);

  return (
    <Drawer anchor="left" open={open} onClose={onClose}>
      <Box sx={{ width: 280, p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <History /> Chat Sessions
        </Typography>
        <Divider />
        <List>
          {sessions.length === 0 ? (
            <Typography variant="body2" sx={{ p: 2, color: 'gray' }}>No history yet.</Typography>
          ) : (
            sessions.map((session) => (
              <ListItem disablePadding key={session.id}>
                <ListItemButton onClick={() => onSelectSession(session.id)}>
                  <ListItemText 
                    primary={session.title} 
                    secondary={new Date(session.created_at).toLocaleDateString()} 
                    primaryTypographyProps={{ noWrap: true, fontWeight: 'bold' }}
                  />
                </ListItemButton>
              </ListItem>
            ))
          )}
        </List>
      </Box>
    </Drawer>
  );
};

// --- PROTECTED ROUTE COMPONENT ---
const PrivateRoute = ({ children, adminOnly = false }) => {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" />;
  if (adminOnly && user.role !== 'admin') return <Navigate to="/" />;
  return children;
};

// --- MAIN LAYOUT ---
function AppContent() {
  const [mode, setMode] = useState('dark');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState(null);
  
  const { user, logout } = useAuth();
  const location = useLocation();

  const theme = React.useMemo(() => createTheme({
    palette: {
      mode,
      primary: { main: mode === 'light' ? '#3F51B5' : '#7986CB' },
      background: { default: mode === 'dark' ? '#0f172a' : '#f5f5f5', paper: mode === 'dark' ? '#1e293b' : '#ffffff' },
    },
  }), [mode]);

  const isLoginPage = location.pathname === '/login';

  const handleSessionSelect = (id) => {
    setSelectedSessionId(id);
    setSidebarOpen(false);
  };

  // Helper for dynamic colors
  const navColor = mode === 'dark' ? '#ffffff' : '#000000';
  const navBg = mode === 'dark' ? '#0f172a' : '#f5f5f5';

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className={`App ${mode === 'dark' ? 'theme-dark' : 'theme-light'}`}>
        
        {/* DYNAMIC TOP NAVBAR - BLENDS WITH BOTH MODES */}
        {!isLoginPage && (
          <AppBar 
            position="sticky" 
            elevation={0} 
            sx={{ 
              top: 0, 
              zIndex: 1200, 
              bgcolor: `${navBg} !important`, 
              borderBottom: `1px solid ${mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'} !important`,
              color: `${navColor} !important`
            }}
          >
            <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
              
              {/* LEFT SIDE: Hamburger & Title */}
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                {user && (
                  <IconButton 
                    edge="start" 
                    onClick={() => setSidebarOpen(true)} 
                    sx={{ mr: 2, color: `${navColor} !important` }}
                  >
                    <MenuIcon />
                  </IconButton>
                )}
                <Typography variant="h6" sx={{ fontWeight: 'bold', letterSpacing: '1px', color: `${navColor} !important` }}>
                  CONFID.AI
                </Typography>
              </Box>

              {/* RIGHT SIDE: Controls */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                
                {/* Light/Dark Toggle */}
                <IconButton onClick={() => setMode(prev => prev === 'light' ? 'dark' : 'light')} sx={{ color: `${navColor} !important` }}>
                  {mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
                </IconButton>

                {user ? (
                  <>
                    <IconButton sx={{ color: `${navColor} !important` }}>
                      <Settings fontSize="small" />
                    </IconButton>

                    <Button 
                      component={Link} 
                      to="/" 
                      onClick={() => setSelectedSessionId(null)} 
                      startIcon={<ChatBubbleOutline fontSize="small" />} 
                      sx={{ color: `${navColor} !important`, fontWeight: 'bold', fontSize: '0.85rem' }}
                    >
                      NEW CHAT
                    </Button>

                    {user.role === 'admin' && (
                      <Button 
                        component={Link} 
                        to="/admin" 
                        startIcon={<AdminPanelSettings fontSize="small" />} 
                        sx={{ color: `${navColor} !important`, fontWeight: 'bold', fontSize: '0.85rem' }}
                      >
                        ADMIN
                      </Button>
                    )}

                    <Button 
                      onClick={logout} 
                      startIcon={<Logout fontSize="small" />} 
                      sx={{ color: `${navColor} !important`, fontWeight: 'bold', fontSize: '0.85rem' }}
                    >
                      LOGOUT
                    </Button>
                  </>
                ) : (
                  <Button component={Link} to="/login" sx={{ color: `${navColor} !important`, fontWeight: 'bold' }}>LOGIN</Button>
                )}
              </Box>

            </Toolbar>
          </AppBar>
        )}

        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} onSelectSession={handleSessionSelect} />

        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={
            <PrivateRoute>
              <QAPage selectedSessionId={selectedSessionId} clearSelection={() => setSelectedSessionId(null)} />
            </PrivateRoute>
          } />
          <Route path="/admin" element={
            <PrivateRoute adminOnly={true}>
              <AdminPage />
            </PrivateRoute>
          } />
        </Routes>

      </div>
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