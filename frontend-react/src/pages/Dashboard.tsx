import { useNavigate } from 'react-router-dom';
import StatsCard from '../components/StatsCard';
import Button from '../components/Button';
import { getAnalyses } from '../utils/storage';
import { useTheme } from '../contexts/ThemeContext';
import { Activity } from 'lucide-react';

export default function Dashboard() {
  const navigate = useNavigate();
  const { isDark } = useTheme();
  const analyses = getAnalyses();

  const total = analyses.length;
  const benign = analyses.filter(a => a.status === 'Benign').length;
  const malignant = analyses.filter(a => a.status === 'Malignant').length;
  const intermediate = analyses.filter(a => a.status === 'Intermediate').length;

  const benignPercent = total > 0 ? ((benign / total) * 100).toFixed(0) : 0;
  const malignantPercent = total > 0 ? ((malignant / total) * 100).toFixed(0) : 0;

  return (
    <div>
      <div className="flex items-center gap-3 mb-8">
        <Activity size={32} className="text-blue-500" />
        <h1 className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Ki-67 Analysis Dashboard
        </h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <StatsCard title="Total Analyses" value={total} color="blue" />
        <StatsCard title="Benign" value={`${benign} (${benignPercent}%)`} color="green" />
        <StatsCard title="Malignant" value={`${malignant} (${malignantPercent}%)`} color="red" />
      </div>

      {intermediate > 0 && (
        <div className="mb-8">
          <StatsCard title="Intermediate Risk" value={intermediate} color="blue" />
        </div>
      )}

      <div className={`rounded-lg border-2 p-6 ${
        isDark
          ? 'bg-gray-800 border-gray-700'
          : 'bg-white border-gray-200'
      }`}>
        <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Quick Actions
        </h2>
        <div className="flex flex-wrap gap-4">
          <Button onClick={() => navigate('/analysis')}>New Analysis</Button>
          <Button onClick={() => navigate('/batch')} variant="secondary">Batch Process</Button>
          <Button onClick={() => navigate('/history')} variant="secondary">View History</Button>
        </div>
      </div>

      {analyses.length > 0 && (
        <div className={`mt-8 rounded-lg border-2 p-6 ${
          isDark
            ? 'bg-gray-800 border-gray-700'
            : 'bg-white border-gray-200'
        }`}>
          <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Recent Analyses
          </h2>
          <div className="space-y-3">
            {analyses.slice(0, 5).map(a => (
              <div key={a.id} className={`flex justify-between items-center py-3 border-b last:border-0 ${
                isDark ? 'border-gray-700' : 'border-gray-200'
              }`}>
                <div>
                  <p className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {a.patientId}
                  </p>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    {a.date}
                  </p>
                </div>
                <div className="text-right">
                  <p className={`font-bold text-lg ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {a.ki67Index.toFixed(1)}%
                  </p>
                  <span className={`text-sm px-3 py-1 rounded-full ${
                    a.status === 'Benign' ? 'bg-green-100 text-green-800' :
                    a.status === 'Malignant' ? 'bg-red-100 text-red-800' :
                    'bg-yellow-100 text-yellow-800'
                  }`}>
                    {a.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
