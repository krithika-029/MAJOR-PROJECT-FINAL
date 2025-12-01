import { useState } from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import Button from '../components/Button';
import ReportGenerator from '../components/ReportGenerator';
import { getAnalyses, clearHistory } from '../utils/storage';
import { useTheme } from '../contexts/ThemeContext';
import { Trash2 } from 'lucide-react';

ChartJS.register(ArcElement, Tooltip, Legend);
// Improve chart sharpness on high-DPI displays
ChartJS.defaults.devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2);

export default function History() {
  const { isDark } = useTheme();
  const [analyses, setAnalyses] = useState(getAnalyses());
  const [searchTerm, setSearchTerm] = useState('');

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all history? This cannot be undone.')) {
      clearHistory();
      setAnalyses([]);
    }
  };

  const filteredAnalyses = analyses.filter(a =>
    a.patientId.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const benignCount = analyses.filter(a => a.status === 'Benign').length;
  const intermediateCount = analyses.filter(a => a.status === 'Intermediate').length;
  const malignantCount = analyses.filter(a => a.status === 'Malignant').length;

  const chartData = {
    labels: ['Benign', 'Intermediate', 'Malignant'],
    datasets: [
      {
        data: [benignCount, intermediateCount, malignantCount],
        backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
        borderColor: ['#059669', '#d97706', '#dc2626'],
        borderWidth: 2,
      },
    ],
  };

  const chartOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    devicePixelRatio: Math.max(2, window.devicePixelRatio || 1),
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: {
          color: isDark ? '#d1d5db' : '#374151',
          // Use round markers for clearer legend visuals
          usePointStyle: true,
          pointStyle: 'circle',
        },
      },
    },
    animation: false,
    layout: { padding: 0 },
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Analysis History
        </h1>
        {analyses.length > 0 && (
          <Button onClick={handleClearHistory} variant="danger">
            <Trash2 size={18} className="inline mr-2" />
            Clear All History
          </Button>
        )}
      </div>

      {analyses.length > 0 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            <div className={`lg:col-span-2 rounded-lg border-2 p-6 ${
              isDark
                ? 'bg-gray-800 border-gray-700'
                : 'bg-white border-gray-200'
            }`}>
              <div className="flex justify-between items-center mb-4">
                <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  All Analyses ({filteredAnalyses.length})
                </h2>
                <input
                  type="text"
                  placeholder="Search Patient ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className={`px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent w-64 ${
                    isDark
                      ? 'bg-gray-700 border-gray-600 text-white'
                      : 'bg-white border-gray-300 text-gray-900'
                  }`}
                />
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className={`border-b-2 ${isDark ? 'border-gray-700' : 'border-gray-300'}`}>
                      <th className={`text-left py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Date</th>
                      <th className={`text-left py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Patient ID</th>
                      <th className={`text-right py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Ki-67</th>
                      <th className={`text-center py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Status</th>
                      <th className={`text-center py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Report</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredAnalyses.map(a => (
                      <tr key={a.id} className={`border-b ${
                        isDark
                          ? 'border-gray-700 hover:bg-gray-700'
                          : 'border-gray-200 hover:bg-gray-50'
                      }`}>
                        <td className={`py-3 px-2 text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                          {a.date}
                        </td>
                        <td className={`py-3 px-2 font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                          {a.patientId}
                        </td>
                        <td className={`py-3 px-2 text-right font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                          {a.ki67Index.toFixed(1)}%
                        </td>
                        <td className="py-3 px-2 text-center">
                          <span className={`px-3 py-1 rounded-full text-sm ${
                            a.status === 'Benign' ? 'bg-green-100 text-green-800' :
                            a.status === 'Malignant' ? 'bg-red-100 text-red-800' :
                            'bg-yellow-100 text-yellow-800'
                          }`}>
                            {a.status}
                          </span>
                        </td>
                        <td className="py-3 px-2 text-center">
                          <ReportGenerator analysis={a} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className={`rounded-lg border-2 p-6 ${
              isDark
                ? 'bg-gray-800 border-gray-700'
                : 'bg-white border-gray-200'
            }`}>
              <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Distribution
              </h2>
              <div style={{ height: '300px', width: '100%' }}>
                <Pie data={chartData} options={chartOptions} />
              </div>
              <div className="mt-6 space-y-2">
                <div className="flex justify-between">
                  <span className={isDark ? 'text-gray-300' : 'text-gray-700'}>Total Analyses:</span>
                  <span className={`font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {analyses.length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-green-600">Benign:</span>
                  <span className="font-bold text-green-600">
                    {benignCount} ({((benignCount/analyses.length)*100).toFixed(0)}%)
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-yellow-600">Intermediate:</span>
                  <span className="font-bold text-yellow-600">
                    {intermediateCount} ({((intermediateCount/analyses.length)*100).toFixed(0)}%)
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-red-600">Malignant:</span>
                  <span className="font-bold text-red-600">
                    {malignantCount} ({((malignantCount/analyses.length)*100).toFixed(0)}%)
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {analyses.length === 0 && (
        <div className={`border-2 border-dashed rounded-lg p-12 text-center ${
          isDark
            ? 'bg-gray-800 border-gray-700'
            : 'bg-gray-50 border-gray-300'
        }`}>
          <p className={`text-lg ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
            No analyses found. Start by creating a new analysis.
          </p>
        </div>
      )}
    </div>
  );
}
