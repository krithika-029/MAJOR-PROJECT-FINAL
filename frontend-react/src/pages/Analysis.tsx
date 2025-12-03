import { useState, useEffect } from 'react';
import FileUpload from '../components/FileUpload';
import Button from '../components/Button';
import ResultsCard from '../components/ResultsCard';
import ClassificationBadge from '../components/ClassificationBadge';
import ReportGenerator from '../components/ReportGenerator';
import ImageVisualization from '../components/ImageVisualization';
import { saveAnalysis, generateId } from '../utils/storage';
import { analyzeImage, AnalysisResult, checkApiHealth } from '../utils/api';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import { FileImage, AlertTriangle } from 'lucide-react';

// Session storage keys
const STORAGE_KEYS = {
  PATIENT_ID: 'ki67_current_patient_id',
  MANUAL_POSITIVE: 'ki67_current_manual_positive',
  MANUAL_NEGATIVE: 'ki67_current_manual_negative',
  NOTES: 'ki67_current_notes',
  RESULTS: 'ki67_current_results',
  FILE_DATA: 'ki67_current_file_data',
  FILE_NAME: 'ki67_current_file_name',
};

export default function Analysis() {
  const navigate = useNavigate();
  const { isDark } = useTheme();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState('');
  const [manualPositive, setManualPositive] = useState('');
  const [manualNegative, setManualNegative] = useState('');
  const [notes, setNotes] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [savedAnalysis, setSavedAnalysis] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiAvailable, setApiAvailable] = useState(true);

  // Restore data from sessionStorage when component mounts
  useEffect(() => {
    const restoreData = async () => {
      try {
        // Restore form data
        const savedPatientId = sessionStorage.getItem(STORAGE_KEYS.PATIENT_ID);
        const savedManualPositive = sessionStorage.getItem(STORAGE_KEYS.MANUAL_POSITIVE);
        const savedManualNegative = sessionStorage.getItem(STORAGE_KEYS.MANUAL_NEGATIVE);
        const savedNotes = sessionStorage.getItem(STORAGE_KEYS.NOTES);
        const savedResults = sessionStorage.getItem(STORAGE_KEYS.RESULTS);
        const savedFileData = sessionStorage.getItem(STORAGE_KEYS.FILE_DATA);
        const savedFileName = sessionStorage.getItem(STORAGE_KEYS.FILE_NAME);

        if (savedPatientId) setPatientId(savedPatientId);
        if (savedManualPositive) setManualPositive(savedManualPositive);
        if (savedManualNegative) setManualNegative(savedManualNegative);
        if (savedNotes) setNotes(savedNotes);
        if (savedResults) setResults(JSON.parse(savedResults));

        // Restore uploaded file
        if (savedFileData && savedFileName) {
          const byteString = atob(savedFileData.split(',')[1]);
          const mimeString = savedFileData.split(',')[0].split(':')[1].split(';')[0];
          const ab = new ArrayBuffer(byteString.length);
          const ia = new Uint8Array(ab);
          for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
          }
          const blob = new Blob([ab], { type: mimeString });
          const file = new File([blob], savedFileName, { type: mimeString });
          setSelectedFile(file);
        }
      } catch (error) {
        console.error('Error restoring session data:', error);
      }
    };

    restoreData();
    checkApiHealth().then(setApiAvailable);
  }, []);

  // Save data to sessionStorage whenever it changes
  useEffect(() => {
    if (patientId) sessionStorage.setItem(STORAGE_KEYS.PATIENT_ID, patientId);
  }, [patientId]);

  useEffect(() => {
    if (manualPositive) sessionStorage.setItem(STORAGE_KEYS.MANUAL_POSITIVE, manualPositive);
  }, [manualPositive]);

  useEffect(() => {
    if (manualNegative) sessionStorage.setItem(STORAGE_KEYS.MANUAL_NEGATIVE, manualNegative);
  }, [manualNegative]);

  useEffect(() => {
    if (notes) sessionStorage.setItem(STORAGE_KEYS.NOTES, notes);
  }, [notes]);

  useEffect(() => {
    if (results) sessionStorage.setItem(STORAGE_KEYS.RESULTS, JSON.stringify(results));
  }, [results]);

  useEffect(() => {
    if (selectedFile) {
      const reader = new FileReader();
      reader.onloadend = () => {
        sessionStorage.setItem(STORAGE_KEYS.FILE_DATA, reader.result as string);
        sessionStorage.setItem(STORAGE_KEYS.FILE_NAME, selectedFile.name);
      };
      reader.readAsDataURL(selectedFile);
    }
  }, [selectedFile]);

  const handleAnalyze = async () => {
    if (!selectedFile) {
      alert('Please select an image file');
      return;
    }
    
    if (!patientId) {
      alert('Please enter a Patient ID');
      return;
    }

    if (!apiAvailable) {
      alert('Backend API is not available. Please ensure the Flask server is running.');
      return;
    }

    setAnalyzing(true);
    setError(null);

    try {
      const result = await analyzeImage({
        image: selectedFile,
        patientId,
        manualPositive: manualPositive ? parseInt(manualPositive) : undefined,
        manualNegative: manualNegative ? parseInt(manualNegative) : undefined,
        notes
      });

      console.log('Analysis result:', result);
      console.log('Images data:', result.images);
      setResults(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      console.error('Analysis error:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSave = () => {
    if (!results) return;

    const analysis = {
      id: results.analysis_id || generateId(),
      date: new Date().toISOString().split('T')[0],
      patientId: patientId || results.patient_data?.patient_id || 'Unknown',
      ki67Index: results.results?.ki67_index || 0,
      status: results.results?.diagnosis?.classification || 'Unknown',
      risk: results.results?.diagnosis?.risk || 'Unknown',
      positiveCells: results.results?.positive_cells || 0,
      negativeCells: results.results?.negative_cells || 0,
      totalCells: results.results?.total_cells || 0,
      imageName: selectedFile?.name || 'Unknown',
      notes,
      // Add full results object for enhanced PDF generation
      analysisResult: results,
    };

    saveAnalysis(analysis);
    setSavedAnalysis(analysis);
    alert('Analysis saved to history!');
  };

  const handleClearSession = () => {
    // Clear all session storage
    Object.values(STORAGE_KEYS).forEach(key => sessionStorage.removeItem(key));
    
    // Reset all state
    setSelectedFile(null);
    setPatientId('');
    setManualPositive('');
    setManualNegative('');
    setNotes('');
    setResults(null);
    setSavedAnalysis(null);
    setError(null);
  };

  return (
    <div>
      <h1 className={`text-3xl font-bold mb-8 ${isDark ? 'text-white' : 'text-gray-900'}`}>
        Single Image Analysis
      </h1>

      {/* API Status Indicator */}
      {!apiAvailable && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
          <AlertTriangle className="text-red-500" />
          <div>
            <p className="font-medium text-red-800">Backend API Unavailable</p>
            <p className="text-sm text-red-600">
              Please ensure the Flask server is running on localhost:5000
            </p>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="font-medium text-red-800">Analysis Error</p>
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-6">
          <div>
            <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Upload Image
            </h2>
            <FileUpload onFileSelect={setSelectedFile} />
            {selectedFile && (
              <div className={`mt-4 p-4 border rounded-lg flex items-center gap-3 ${
                isDark
                  ? 'bg-blue-900 border-blue-700'
                  : 'bg-blue-50 border-blue-200'
              }`}>
                <FileImage className="text-blue-500" />
                <div>
                  <p className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {selectedFile.name}
                  </p>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className={`rounded-lg border-2 p-6 ${
            isDark
              ? 'bg-gray-800 border-gray-700'
              : 'bg-white border-gray-200'
          }`}>
            <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Patient Information
            </h2>
            <div className="space-y-4">
              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  isDark ? 'text-gray-200' : 'text-gray-700'
                }`}>
                  Patient ID *
                </label>
                <input
                  type="text"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="e.g., P001"
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                    isDark
                      ? 'bg-gray-700 border-gray-600 text-white'
                      : 'bg-white border-gray-300 text-gray-900'
                  }`}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className={`block text-sm font-medium mb-2 ${
                    isDark ? 'text-gray-200' : 'text-gray-700'
                  }`}>
                    Manual Positive Count
                  </label>
                  <input
                    type="number"
                    value={manualPositive}
                    onChange={(e) => setManualPositive(e.target.value)}
                    placeholder="e.g., 98"
                    className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                      isDark
                        ? 'bg-gray-700 border-gray-600 text-white'
                        : 'bg-white border-gray-300 text-gray-900'
                    }`}
                  />
                </div>
                <div>
                  <label className={`block text-sm font-medium mb-2 ${
                    isDark ? 'text-gray-200' : 'text-gray-700'
                  }`}>
                    Manual Negative Count
                  </label>
                  <input
                    type="number"
                    value={manualNegative}
                    onChange={(e) => setManualNegative(e.target.value)}
                    placeholder="e.g., 85"
                    className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                      isDark
                        ? 'bg-gray-700 border-gray-600 text-white'
                        : 'bg-white border-gray-300 text-gray-900'
                    }`}
                  />
                </div>
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  isDark ? 'text-gray-200' : 'text-gray-700'
                }`}>
                  Clinical Notes
                </label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Optional notes about the sample..."
                  rows={4}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                    isDark
                      ? 'bg-gray-700 border-gray-600 text-white'
                      : 'bg-white border-gray-300 text-gray-900'
                  }`}
                />
              </div>

              <div className="flex gap-3">
                <Button
                  onClick={handleAnalyze}
                  disabled={analyzing || !patientId}
                  className="flex-1"
                >
                  {analyzing ? 'Analyzing...' : 'Analyze'}
                </Button>
                <Button
                  onClick={handleClearSession}
                  variant="secondary"
                  disabled={analyzing}
                  className="flex-shrink-0"
                >
                  Clear Form
                </Button>
              </div>
            </div>
          </div>
        </div>

        <div>
          {results && (
            <div className="space-y-6">
              {/* Visualization Images */}
              <ImageVisualization 
                original={results.images?.original}
                analyzed={results.images?.analyzed}
                coordinates={results.cell_coordinates}
                ki67Index={results.results?.ki67_index}
              />

              <div>
                <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  Results
                </h2>
                <ResultsCard 
                  ki67Index={results.results?.ki67_index || 0}
                  totalCells={results.results?.total_cells || 0}
                  positiveCells={results.results?.positive_cells || 0}
                  negativeCells={results.results?.negative_cells || 0}
                />
              </div>

              <div>
                <ClassificationBadge
                  status={results.results?.diagnosis?.classification || 'Unknown'}
                  risk={results.results?.diagnosis?.risk || 'Unknown Risk'}
                />
              </div>

              <div className="flex gap-4 flex-col">
                {savedAnalysis && (
                  <ReportGenerator analysis={savedAnalysis} />
                )}
                <Button onClick={handleSave}>
                  {savedAnalysis ? 'Already Saved' : 'Save to History'}
                </Button>
                {savedAnalysis && (
                  <Button onClick={() => navigate('/history')} variant="secondary">
                    View in History
                  </Button>
                )}
              </div>
            </div>
          )}

          {!results && !analyzing && (
            <div className={`border-2 border-dashed rounded-lg p-12 text-center ${
              isDark
                ? 'bg-gray-800 border-gray-700'
                : 'bg-gray-50 border-gray-300'
            }`}>
              <p className={isDark ? 'text-gray-400' : 'text-gray-500'}>
                Results will appear here after analysis
              </p>
            </div>
          )}

          {analyzing && (
            <div className={`border-2 rounded-lg p-12 text-center ${
              isDark
                ? 'bg-blue-900 border-blue-700'
                : 'bg-blue-50 border-blue-200'
            }`}>
              <div className="animate-pulse">
                <div className={`h-4 rounded w-3/4 mx-auto mb-4 ${
                  isDark ? 'bg-blue-700' : 'bg-blue-200'
                }`}></div>
                <div className={`h-4 rounded w-1/2 mx-auto ${
                  isDark ? 'bg-blue-700' : 'bg-blue-200'
                }`}></div>
              </div>
              <p className={`mt-4 font-medium ${isDark ? 'text-blue-200' : 'text-blue-700'}`}>
                Analyzing image...
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
