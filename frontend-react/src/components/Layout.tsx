import { Link, useLocation, useNavigate } from 'react-router-dom';
import { LayoutDashboard, FileImage, FolderOpen, History, Moon, Sun, LogOut } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const { isDark, toggleDark } = useTheme();
  const { user, signOut } = useAuth();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/analysis', label: 'Analysis', icon: FileImage },
    { path: '/batch', label: 'Batch', icon: FolderOpen },
    { path: '/history', label: 'History', icon: History },
  ];

  const handleSignOut = async () => {
    await signOut();
    navigate('/login');
  };

  return (
    <div className={`min-h-screen ${isDark ? 'bg-gray-900' : 'bg-gray-100'}`}>
      <nav className={`${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b-2`}>
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <div className={`w-8 h-8 rounded-lg ${isDark ? 'bg-blue-600' : 'bg-blue-500'}`}></div>
              <span className={`font-bold text-xl ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Ki-67 Analyzer
              </span>
            </div>

            <div className="flex items-center gap-1">
              {navItems.map(item => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      isActive
                        ? isDark
                          ? 'bg-blue-600 text-white'
                          : 'bg-blue-500 text-white'
                        : isDark
                          ? 'text-gray-300 hover:bg-gray-700'
                          : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <Icon size={18} />
                    <span className="font-medium">{item.label}</span>
                  </Link>
                );
              })}
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={toggleDark}
                className={`p-2 rounded-lg transition-colors ${
                  isDark
                    ? 'bg-gray-700 text-yellow-400 hover:bg-gray-600'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
                title="Toggle dark mode"
              >
                {isDark ? <Sun size={18} /> : <Moon size={18} />}
              </button>

              {user && (
                <div className={`flex items-center gap-3 ml-4 pl-4 border-l ${
                  isDark ? 'border-gray-700' : 'border-gray-200'
                }`}>
                  <span className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                    {user.email}
                  </span>
                  <button
                    onClick={handleSignOut}
                    className={`p-2 rounded-lg transition-colors ${
                      isDark
                        ? 'text-red-400 hover:bg-gray-700'
                        : 'text-red-500 hover:bg-red-50'
                    }`}
                    title="Sign out"
                  >
                    <LogOut size={18} />
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </nav>

      <main className={`max-w-7xl mx-auto px-6 py-8 ${isDark ? 'text-white' : ''}`}>
        {children}
      </main>
    </div>
  );
}
