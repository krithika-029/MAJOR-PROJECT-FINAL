import { useState, useEffect } from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import Button from '../components/Button';
import { useTheme } from '../contexts/ThemeContext';
import { ChevronLeft, ChevronRight, Download, AlertCircle } from 'lucide-react';

ChartJS.register(ArcElement, Tooltip, Legend);
ChartJS.defaults.devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2);

interface HistoryItem {
  analysis_id: string;
  created_at: string;
  patient_name: string;
  patient_id: string;
  original_filename: string;
  metrics: {
    positive_cells: number;
    negative_cells: number;
    total_cells: number;
    ki67_index: number;
    classification: string;
    risk: string;
    malignant: boolean;
  };
  qc: {
    available: boolean;
    flagged: boolean;
    reason: string;
  };
  reports: {
    pdf: string;
    csv: string;
  };
}

export default function History() {
  const { isDark } = useTheme();
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetchHistory();
  }, [page]);

  const fetchHistory = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`/api/history?page=${page}&page_size=${pageSize}`);
      if (!response.ok) throw new Error('Failed to fetch history');
      const data = await response.json();
      setItems(data.items || []);
      setTotal(data.total || 0);
    } catch (err: any) {
      setError(err.message || 'Error loading history');
    } finally {
      setLoading(false);
    }
  };

  const filteredItems = items.filter(item =>
    (item.patient_id?.toLowerCase() || '').includes(searchTerm.toLowerCase()) ||
    (item.patient_name?.toLowerCase() || '').includes(searchTerm.toLowerCase())
  );

  const benignCount = items.filter(item => !item.metrics.malignant && item.metrics.ki67_index < 14).length;
  const intermediateCount = items.filter(item => !item.metrics.malignant && item.metrics.ki67_index >= 14).length;
  const malignantCount = items.filter(item => item.metrics.malignant).length;

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

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Analysis History
          </h1>
          <p className={`mt-2 text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            {total} total analyses stored in database
          </p>
        </div>
      </div>

      {loading && (
        <div className={`text-center py-12 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          Loading history...
        </div>
      )}

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 flex items-center">
          <AlertCircle size={20} className="mr-2" />
          {error}
        </div>
      )}

      {!loading && !error && items.length > 0 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            <div className={`lg:col-span-2 rounded-lg border-2 p-6 ${
              isDark
                ? 'bg-gray-800 border-gray-700'
                : 'bg-white border-gray-200'
            }`}>
              <div className="flex justify-between items-center mb-4">
                <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  Recent Analyses (Page {page} of {totalPages})
                </h2>
                <input
                  type="text"
                  placeholder="Search Patient ID/Name..."
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
                      <th className={`text-left py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Patient</th>
                      <th className={`text-right py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Ki-67</th>
                      <th className={`text-center py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Classification</th>
                      <th className={`text-center py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>QC</th>
                      <th className={`text-center py-3 px-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Reports</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredItems.map(item => {
                      const dateStr = new Date(item.created_at).toLocaleDateString();
                      const patientDisplay = item.patient_name || item.patient_id || 'N/A';
                      const ki67 = item.metrics.ki67_index;
                      const classification = item.metrics.classification;
                      const isQcFlagged = item.qc.flagged;
                      
                      return (
                        <tr key={item.analysis_id} className={`border-b ${
                          isDark
                            ? 'border-gray-700 hover:bg-gray-700'
                            : 'border-gray-200 hover:bg-gray-50'
                        }`}>
                          <td className={`py-3 px-2 text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                            {dateStr}
                          </td>
                          <td className={`py-3 px-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                            <div className="font-medium">{patientDisplay}</div>
                            {item.patient_id && item.patient_name && (
                              <div className="text-xs text-gray-500">ID: {item.patient_id}</div>
                            )}
                          </td>
                          <td className={`py-3 px-2 text-right font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                            {ki67.toFixed(1)}%
                          </td>
                          <td className="py-3 px-2 text-center">
                            <span className={`px-3 py-1 rounded-full text-sm ${
                              item.metrics.malignant ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                            }`}>
                              {classification}
                            </span>
                          </td>
                          <td className="py-3 px-2 text-center">
                            {isQcFlagged ? (
                              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-xs" title={item.qc.reason}>
                                Flagged
                              </span>
                            ) : (
                              <span className="text-green-600 text-xs">✓ OK</span>
                            )}
                          </td>
                          <td className="py-3 px-2 text-center">
                            <div className="flex gap-2 justify-center">
                              <a
                                href={item.reports.pdf}
                                className="text-blue-600 hover:text-blue-800 text-sm flex items-center"
                                title="Download PDF"
                              >
                                <Download size={16} className="mr-1" /> PDF
                              </a>
                              <a
                                href={item.reports.csv}
                                className="text-blue-600 hover:text-blue-800 text-sm flex items-center"
                                title="Download CSV"
                              >
                                <Download size={16} className="mr-1" /> CSV
                              </a>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              
              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex justify-center items-center gap-4 mt-6">
                  <Button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1}
                    variant="secondary"
                  >
                    <ChevronLeft size={18} /> Previous
                  </Button>
                  <span className={isDark ? 'text-gray-300' : 'text-gray-700'}>
                    Page {page} of {totalPages}
                  </span>
                  <Button
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages}
                    variant="secondary"
                  >
                    Next <ChevronRight size={18} />
                  </Button>
                </div>
              )}
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
                  <span className={isDark ? 'text-gray-300' : 'text-gray-700'}>Current Page:</span>
                  <span className={`font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {items.length} analyses
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-green-600">Benign:</span>
                  <span className="font-bold text-green-600">
                    {benignCount} ({items.length > 0 ? ((benignCount/items.length)*100).toFixed(0) : 0}%)
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-yellow-600">Intermediate:</span>
                  <span className="font-bold text-yellow-600">
                    {intermediateCount} ({items.length > 0 ? ((intermediateCount/items.length)*100).toFixed(0) : 0}%)
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-red-600">Malignant:</span>
                  <span className="font-bold text-red-600">
                    {malignantCount} ({items.length > 0 ? ((malignantCount/items.length)*100).toFixed(0) : 0}%)
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {!loading && !error && items.length === 0 && (
        <div className={`border-2 border-dashed rounded-lg p-12 text-center ${
          isDark
            ? 'bg-gray-800 border-gray-700'
            : 'bg-gray-50 border-gray-300'
        }`}>
          <p className={`text-lg ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
            No analyses found in database. Start by creating a new analysis.
          </p>
        </div>
      )}
    </div>
  );
}
