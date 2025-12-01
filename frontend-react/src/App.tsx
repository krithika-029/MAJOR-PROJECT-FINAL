import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import { useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import Batch from './pages/Batch';
import History from './pages/History';
import Login from './pages/Login';
import { initializeSampleData } from './utils/storage';

function ProtectedRoute({ element }: { element: React.ReactNode }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse">Loading...</div>
      </div>
    );
  }

  return user ? element : <Navigate to="/login" replace />;
}

function App() {
  useEffect(() => {
    initializeSampleData();
  }, []);

  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/*"
            element={
              <ProtectedRoute
                element={
                  <Layout>
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="/analysis" element={<Analysis />} />
                      <Route path="/batch" element={<Batch />} />
                      <Route path="/history" element={<History />} />
                    </Routes>
                  </Layout>
                }
              />
            }
          />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
