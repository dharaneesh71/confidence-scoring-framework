import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Typography, List, ListItemButton, ListItemText, Drawer, useTheme} from '@mui/material';

const Sidebar = ({ sessions, open, onClose }) => {
  const navigate = useNavigate();
  const theme = useTheme();

  const groupSessions = (sessionsList) => {
    const groups = { Today: [], Yesterday: [], "Previous 7 Days": [], "Older": [] };
    
    const now = new Date();
    now.setHours(0, 0, 0, 0); 

    const sorted = [...sessionsList].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    sorted.forEach(s => {
      const dateString = s.created_at.endsWith('Z') ? s.created_at : `${s.created_at}Z`;
      const date = new Date(dateString);
      date.setHours(0, 0, 0, 0); 
      const diffDays = Math.round((now - date) / (1000 * 60 * 60 * 24));
      
      if (diffDays <= 0) groups.Today.push(s);
      else if (diffDays === 1) groups.Yesterday.push(s);
      else if (diffDays <= 7) groups["Previous 7 Days"].push(s);
      else groups["Older"].push(s);
    });
    
    return groups;
  };

  const groupedData = groupSessions(sessions);

  return (
    <Drawer
    variant="temporary"
    anchor="left"
    open={open}
    onClose={onClose}
    ModalProps={{ keepMounted: true }}
    slotProps={{ backdrop: { style: { backgroundColor: 'transparent' } } }}
    PaperProps={{
      style: {
        top: '64px',
        height: 'calc(100% - 64px)',
        width: 280,
        backgroundColor: theme.palette.mode === 'dark' ? '#000000' : '#ffffff',  // ✅ mode-aware
        backgroundImage: 'none',
        color: theme.palette.mode === 'dark' ? 'white' : '#0f172a',              // ✅ mode-aware
        boxSizing: 'border-box',
        borderRight: theme.palette.mode === 'light' ? '1px solid rgba(0,0,0,0.1)' : 'none',
      }
    }}
    >
      <Box style={{ width: 280, backgroundColor: theme.palette.mode === 'dark' ? '#000000' : '#ffffff', color: theme.palette.mode === 'dark' ? 'white' : '#0f172a', height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box style={{ padding: '16px', borderBottom: theme.palette.mode === 'dark' ? '1px solid #1e293b' : '1px solid rgba(0,0,0,0.1)' }}>
          <Typography variant="h6" sx={{ fontWeight: 'bold', color:  '#a78bfa'  }}>CHAT HISTORY</Typography>
        </Box>
        
        <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 1 }}>
          {sessions.length === 0 ? (
            <Typography sx={{ p: 2, color: 'gray', textAlign: 'center' }}>No history yet.</Typography>
          ) : (
            Object.entries(groupedData).map(([label, items]) => (
              items.length > 0 && (
                <Box key={label} sx={{ mt: 1, mb: 2 }}>
                  <Typography variant="caption" sx={{ px: 2, color: '#64748b', fontWeight: 'bold', textTransform: 'uppercase' }}>
                    {label}
                  </Typography>
                  <List disablePadding>
                    {items.map((session) => (
                      <ListItemButton
                        key={session.id}
                        onClick={() => {
                          navigate(`/chat/${session.id}`);
                          onClose();
                        }}
                        sx={{ borderRadius: 1, mx: 1, color: 'white', '&:hover': { bgcolor: 'rgba(0, 209, 255, 0.1)' } }}
                      >
                        <ListItemText
                          primary={session.title || "New Chat"}
                          primaryTypographyProps={{ noWrap: true, fontSize: '0.9rem', color: theme.palette.mode === 'dark' ? 'white' : '#0f172a' }}
                        />
                      </ListItemButton>
                    ))}
                  </List>
                </Box>
              )
            ))
          )}
        </Box>
      </Box>
    </Drawer>
  );
};

export default Sidebar;
