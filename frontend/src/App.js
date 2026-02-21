import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import {
  AppBar, Toolbar, Typography, Button, Box,
  CssBaseline, ThemeProvider, createTheme, IconButton, Drawer, List, ListItem, ListItemText, Divider, ListItemButton
} from '@mui/material';
import { 
  QuestionAnswer, AdminPanelSettings, Brightness4, Brightness7, 
  Logout, History, Menu as MenuIcon 
} from '@mui/icons-material';

import QAPage from './pages/QAPage';
import AdminPage from './pages/AdminPage';
import LoginPage from './pages/LoginPage';
import { AuthProvider, useAuth } from './context/AuthContext';
import './App.css';

// --- SIDEBAR COMPONENT ---
const Sidebar = ({ open, onClose, onSelectHistory }) => {
  const [history, setHistory] = useState([]);
  const { user } = useAuth();

  useEffect(() => {
    if (open && user) {
      fetch('http://localhost:8000/api/history', {
        headers: { Authorization: `Bearer ${user.token}` }
      })
      .then(res => res.json())
      .then(data => setHistory(data))
      .catch(err => console.error(err));
    }
  }, [open, user]);

  return (
    <Drawer anchor="left" open={open} onClose={onClose}>
      <Box sx={{ width: 280, p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <History /> History
        </Typography>
        <Divider />
        <List>
          {history.length === 0 ? (
            <Typography variant="body2" sx={{ p: 2, color: 'gray' }}>No history yet.</Typography>
          ) : (
            history.map((chat) => (
              <ListItem disablePadding key={chat.id}>
                {/* MODIFIED: Now clicking an item triggers the onSelectHistory function */}
                <ListItemButton onClick={() => onSelectHistory(chat.id)}>
                  <ListItemText 
                    primary={chat.question} 
                    secondary={new Date(chat.timestamp).toLocaleDateString()} 
                    primaryTypographyProps={{ noWrap: true }}
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
  
  // NEW: State to track which history item the user clicked in the sidebar
  const [selectedHistoryId, setSelectedHistoryId] = useState(null);
  
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

  // NEW: Function to handle when a user clicks a history item
  const handleHistorySelect = (id) => {
    setSelectedHistoryId(id);
    setSidebarOpen(false); // Close the sidebar after selection
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className={`App ${mode === 'dark' ? 'theme-dark' : 'theme-light'}`}>
        
        {!isLoginPage && (
          <AppBar position="static" color="transparent" elevation={0} sx={{ backdropFilter: 'blur(10px)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
            <Toolbar>
              {user && (
                <IconButton edge="start" color="inherit" onClick={() => setSidebarOpen(true)} sx={{ mr: 2 }}>
                  <MenuIcon />
                </IconButton>
              )}
              
              <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
                CONFID.AI
              </Typography>

              <IconButton onClick={() => setMode(prev => prev === 'light' ? 'dark' : 'light')} color="inherit">
                {mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
              </IconButton>

              {user ? (
                <>
                  <Button component={Link} to="/" startIcon={<QuestionAnswer />} color="inherit" onClick={() => setSelectedHistoryId(null)}>Q&A</Button>
                  {user.role === 'admin' && (
                    <Button component={Link} to="/admin" startIcon={<AdminPanelSettings />} color="inherit">Admin</Button>
                  )}
                  <Button onClick={logout} startIcon={<Logout />} color="inherit" sx={{ ml: 2 }}>Logout</Button>
                </>
              ) : (
                <Button component={Link} to="/login" color="inherit">Login</Button>
              )}
            </Toolbar>
          </AppBar>
        )}

        {/* Sidebar now passes the selection back up */}
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} onSelectHistory={handleHistorySelect} />

        <Routes>
          <Route path="/login" element={<LoginPage />} />
          
          <Route path="/" element={
            <PrivateRoute>
              {/* Pass the selected ID down to the QAPage so it knows what to load */}
              <QAPage selectedHistoryId={selectedHistoryId} clearSelection={() => setSelectedHistoryId(null)} />
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