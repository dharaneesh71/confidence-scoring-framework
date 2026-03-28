import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import {
  AppBar, Toolbar, Typography, Button, Box,
  CssBaseline, ThemeProvider, createTheme, IconButton
} from '@mui/material';
import { 
  AdminPanelSettings, Brightness4, Brightness7, 
  Logout, Settings, ChatBubbleOutline, Menu as MenuIcon 
} from '@mui/icons-material';

import QAPage from './pages/QAPage';
import AdminPage from './pages/AdminPage';
import LoginPage from './pages/LoginPage';
import Sidebar from './components/Sidebar';
import { AuthProvider, useAuth } from './context/AuthContext';
import './App.css';

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
      background: { default: mode === 'dark' ? '#0f172a' : '#f5f5f5', paper: mode === 'dark' ? '#1e293b' : '#ffffff' },
    },
  }), [mode]);

  const isLoginPage = location.pathname === '/login';

  useEffect(() => {
    if (user && !isLoginPage) {
      fetch('http://localhost:8000/api/history', {
        headers: { Authorization: `Bearer ${user.token}` }
      })
      .then(res => res.json())
      .then(data => setSessions(data))
      .catch(err => console.error(err));
    }
  }, [user, isLoginPage, location.pathname]); 

  const navColor = mode === 'dark' ? '#ffffff' : '#000000';
  const navBg = mode === 'dark' ? '#0f172a' : '#f5f5f5';

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {/* FIX: Changed minHeight to height, added overflow hidden */}
      <div className={`App ${mode === 'dark' ? 'theme-dark' : 'theme-light'}`} style={{ display: 'flex', height: '100vh', overflow: 'hidden', flexDirection: 'column' }}>
        
        {!isLoginPage && user && (
           <Sidebar 
             sessions={sessions} 
             isOpen={sidebarOpen} 
             onToggle={() => setSidebarOpen(false)} 
           />
        )}

        {!isLoginPage && (
          <AppBar 
            position="sticky" 
            elevation={0} 
            sx={{ 
              top: 0, 
              zIndex: 1200, 
              bgcolor: `${navBg} !important`, 
              borderBottom: `1px solid ${mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'} !important`,
              color: `${navColor} !important`,
              flexShrink: 0 // FIX: Stops navbar from shrinking
            }}
          >
            <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}> 
                {user && (
                  <IconButton edge="start" onClick={() => setSidebarOpen(true)} sx={{ mr: 2, color: `${navColor} !important` }}>
                    <MenuIcon />
                  </IconButton>
                )}
                <Typography variant="h6" sx={{ fontWeight: 'bold', letterSpacing: '1px', color: `${navColor} !important` }}>
                  CONFID.AI
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <IconButton onClick={() => setMode(prev => prev === 'light' ? 'dark' : 'light')} sx={{ color: `${navColor} !important` }}>
                  {mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
                </IconButton>

                {user ? (
                  <>
                    <IconButton sx={{ color: `${navColor} !important`, display: { xs: 'none', sm: 'inline-flex' } }}>
                      <Settings fontSize="small" />
                    </IconButton>

                    <Button component={Link} to="/chat" startIcon={<ChatBubbleOutline fontSize="small" />} sx={{ color: `${navColor} !important`, fontWeight: 'bold', fontSize: '0.85rem', display: { xs: 'none', sm: 'inline-flex' } }}>
                      NEW CHAT
                    </Button>

                    {user.role === 'admin' && (
                      <Button component={Link} to="/admin" startIcon={<AdminPanelSettings fontSize="small" />} sx={{ color: `${navColor} !important`, fontWeight: 'bold', fontSize: '0.85rem' }}>
                        ADMIN
                      </Button>
                    )}

                    <Button onClick={logout} startIcon={<Logout fontSize="small" />} sx={{ color: `${navColor} !important`, fontWeight: 'bold', fontSize: '0.85rem' }}>
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

        {/* FIX: Ensure the routing container takes the rest of the height and hides overflow */}
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/" element={<Navigate to="/chat" />} />
            <Route path="/chat" element={
              <PrivateRoute>
                <QAPage />
              </PrivateRoute>
            } />
            <Route path="/chat/:sessionId" element={
              <PrivateRoute>
                <QAPage />
              </PrivateRoute>
            } />
            <Route path="/admin" element={
              <PrivateRoute adminOnly={true}>
                <AdminPage />
              </PrivateRoute>
            } />
          </Routes>
        </Box>
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