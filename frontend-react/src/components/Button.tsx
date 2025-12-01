import { useTheme } from '../contexts/ThemeContext';

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
  type?: 'button' | 'submit';
  disabled?: boolean;
}

export default function Button({
  children,
  onClick,
  variant = 'primary',
  type = 'button',
  disabled = false
}: ButtonProps) {
  const { isDark } = useTheme();

  const variants = {
    primary: 'bg-blue-500 hover:bg-blue-600 text-white',
    secondary: isDark
      ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
      : 'bg-gray-200 hover:bg-gray-300 text-gray-800',
    danger: 'bg-red-500 hover:bg-red-600 text-white',
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`px-6 py-2.5 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${variants[variant]}`}
    >
      {children}
    </button>
  );
}
