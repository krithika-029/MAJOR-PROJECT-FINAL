import { useTheme } from '../contexts/ThemeContext';

interface StatsCardProps {
  title: string;
  value: string | number;
  color?: string;
}

export default function StatsCard({ title, value, color = 'blue' }: StatsCardProps) {
  const { isDark } = useTheme();

  const colorClasses = {
    blue: isDark ? 'bg-blue-900 border-blue-700' : 'bg-blue-50 border-blue-200',
    green: isDark ? 'bg-green-900 border-green-700' : 'bg-green-50 border-green-200',
    red: isDark ? 'bg-red-900 border-red-700' : 'bg-red-50 border-red-200',
  };

  const textColor = {
    blue: isDark ? 'text-blue-300' : 'text-gray-600',
    green: isDark ? 'text-green-300' : 'text-gray-600',
    red: isDark ? 'text-red-300' : 'text-gray-600',
  };

  return (
    <div className={`p-6 rounded-lg border-2 ${colorClasses[color as keyof typeof colorClasses] || colorClasses.blue}`}>
      <p className={`text-sm font-medium mb-2 ${textColor[color as keyof typeof textColor] || textColor.blue}`}>
        {title}
      </p>
      <p className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
        {value}
      </p>
    </div>
  );
}
