import { useState } from 'react';
import Button from '../components/Button';
import { saveAnalysis, generateId } from '../utils/storage';
import { analyzeBatch, AnalysisResult, checkApiHealth } from '../utils/api';
import { useTheme } from '../contexts/ThemeContext';
import { X, FileImage, AlertTriangle } from 'lucide-react';
import ImageVisualization from '../components/ImageVisualization';
import ResultsCard from '../components/ResultsCard';
import ClassificationBadge from '../components/ClassificationBadge';

interface BatchFile {
  id: string;
  file: File;
  processed: boolean;
  result?: AnalysisResult;
}

export default function Batch() {
  const { isDark } = useTheme();
  const [files, setFiles] = useState<BatchFile[]>([]);
  const [patientPrefix, setPatientPrefix] = useState('PT-BATCH');
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [apiAvailable, setApiAvailable] = useState(true);

  // Check API availability
  useState(() => {
    checkApiHealth().then(setApiAvailable);
  });

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    const newFiles = selectedFiles.map(file => ({
      id: generateId(),
      file,
      processed: false,
    }));
    setFiles([...files, ...newFiles]);
  };

  const removeFile = (id: string) => {
    setFiles(files.filter(f => f.id !== id));
  };

  const processAll = async () => {
    if (!apiAvailable) {
      alert('Backend API is not available. Please ensure the Flask server is running.');
      return;
    }

    if (files.length === 0) {
      alert('Please select some images first');
      return;
    }

    setProcessing(true);
    setProgress(0);
    setError(null);

    try {
      const fileList = files.map(f => f.file);
      const batchResult = await analyzeBatch({
        images: fileList,
        patientPrefix
      });

      // Update files with results
      const processedFiles = files.map((file, index) => ({
        ...file,
        processed: true,
        result: batchResult.results[index]
      }));

      setFiles(processedFiles);
      setResults(batchResult.results);
      setProgress(100);

      // Save to local storage
      batchResult.results.forEach((result, index) => {
        const analysis = {
          id: result.analysis_id || result.id,
          date: new Date().toISOString().split('T')[0],
          patientId: result.patient_data?.patient_id || `BATCH-${index+1}`,
          ki67Index: result.results?.ki67_index || 0,
          status: result.results?.diagnosis?.classification || 'Unknown',
          risk: result.results?.diagnosis?.risk || 'Unknown',
          positiveCells: result.results?.positive_cells || 0,
          negativeCells: result.results?.negative_cells || 0,
          totalCells: result.results?.total_cells || 0,
          imageName: result.batch_info?.original_filename || 'Unknown',
          notes: result.patient_data?.clinical_notes || '',
        };
        saveAnalysis(analysis);
      });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch analysis failed');
      console.error('Batch analysis error:', err);
    } finally {
      setProcessing(false);
    }
  };

  const processedFiles = files.filter(f => f.processed);
  const benignCount = processedFiles.filter(f => f.result?.results?.diagnosis?.classification === 'Benign').length;
  const lowMalignantCount = processedFiles.filter(f => f.result?.results?.diagnosis?.classification === 'Low Malignant Potential').length;
  const borderlineCount = processedFiles.filter(f => f.result?.results?.diagnosis?.classification === 'Borderline Malignant').length;
  const malignantCount = processedFiles.filter(f => f.result?.results?.diagnosis?.classification === 'Malignant').length;
  const avgKi67 = processedFiles.length > 0
    ? (processedFiles.reduce((sum, f) => sum + (f.result?.results?.ki67_index || 0), 0) / processedFiles.length).toFixed(1)
    : 0;

  return (
    <div>
      <h1 className={`text-3xl font-bold mb-8 ${isDark ? 'text-white' : 'text-gray-900'}`}>
        Batch Processing
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
          <p className="font-medium text-red-800">Batch Analysis Error</p>
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Upload Section */}
      <div className={`rounded-lg border-2 p-6 mb-8 ${
        isDark
          ? 'bg-gray-800 border-gray-700'
          : 'bg-white border-gray-200'
      }`}>
        <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Upload Multiple Images
        </h2>
        
        {/* Patient ID Prefix Input */}
        <div className="mb-4">
          <label className={`block text-sm font-medium mb-2 ${
            isDark ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Patient ID Prefix
          </label>
          <input
            type="text"
            value={patientPrefix}
            onChange={(e) => setPatientPrefix(e.target.value)}
            placeholder="e.g., PT-BATCH"
            className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              isDark
                ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
            }`}
          />
          <p className={`mt-1 text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Images will be labeled as: {patientPrefix}-001, {patientPrefix}-002, etc.
          </p>
        </div>

        {/* File Input */}
        <input
          type="file"
          onChange={handleFileSelect}
          accept="image/*"
          multiple
          className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-500 file:text-white hover:file:bg-blue-600 file:cursor-pointer"
        />
      </div>

      {files.length > 0 && (
        <>
          {/* File List */}
          <div className={`rounded-lg border-2 p-6 mb-8 ${
            isDark
              ? 'bg-gray-800 border-gray-700'
              : 'bg-white border-gray-200'
          }`}>
            <div className="flex justify-between items-center mb-4">
              <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Uploaded Files ({files.length})
              </h2>
              <Button
                onClick={processAll}
                disabled={processing || files.length === 0 || !apiAvailable}
              >
                {processing ? 'Processing...' : 'Process All'}
              </Button>
            </div>

            {processing && (
              <div className="mb-4">
                <div className={`rounded-full h-3 overflow-hidden ${isDark ? 'bg-gray-700' : 'bg-gray-200'}`}>
                  <div
                    className="h-full bg-blue-500 transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className={`text-sm mt-1 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  Processing... {progress.toFixed(0)}%
                </p>
              </div>
            )}

            <div className="space-y-2">
              {files.map((batchFile) => (
                <div
                  key={batchFile.id}
                  className={`flex items-center justify-between p-3 rounded-lg border ${
                    batchFile.processed
                      ? isDark
                        ? 'bg-green-900 border-green-700'
                        : 'bg-green-50 border-green-200'
                      : isDark
                      ? 'bg-gray-700 border-gray-600'
                      : 'bg-gray-50 border-gray-200'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <FileImage className={`${
                      batchFile.processed ? 'text-green-500' : 'text-gray-500'
                    }`} />
                    <div>
                      <p className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                        {batchFile.file.name}
                      </p>
                      {batchFile.processed && batchFile.result && (
                        <p className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                          Ki-67 Index: {batchFile.result.results?.ki67_index?.toFixed(1)}% | {batchFile.result.results?.diagnosis?.classification}
                        </p>
                      )}
                    </div>
                  </div>
                  {!processing && (
                    <button
                      onClick={() => removeFile(batchFile.id)}
                      className="text-red-500 hover:bg-red-100 p-2 rounded"
                    >
                      <X size={16} />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Results Summary */}
          {processedFiles.length > 0 && (
            <div className={`rounded-lg border-2 p-6 ${
              isDark
                ? 'bg-gray-800 border-gray-700'
                : 'bg-white border-gray-200'
            }`}>
              <h2 className={`text-xl font-bold mb-6 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Batch Analysis Results
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className={`p-4 rounded-lg border ${
                  isDark
                    ? 'bg-gray-700 border-gray-600'
                    : 'bg-gray-50 border-gray-200'
                }`}>
                  <p className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {processedFiles.length}
                  </p>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Total Processed
                  </p>
                </div>
                
                <div className={`p-4 rounded-lg border ${
                  isDark
                    ? 'bg-green-900 border-green-700'
                    : 'bg-green-50 border-green-200'
                }`}>
                  <p className="text-2xl font-bold text-green-600">{benignCount}</p>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Benign
                  </p>
                </div>
                
                <div className={`p-4 rounded-lg border ${
                  isDark
                    ? 'bg-yellow-900 border-yellow-700'
                    : 'bg-yellow-50 border-yellow-200'
                }`}>
                  <p className="text-2xl font-bold text-yellow-600">{lowMalignantCount}</p>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Low Malignant
                  </p>
                </div>
                
                <div className={`p-4 rounded-lg border ${
                  isDark
                    ? 'bg-red-900 border-red-700'
                    : 'bg-red-50 border-red-200'
                }`}>
                  <p className="text-2xl font-bold text-orange-600">{borderlineCount}</p>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Borderline
                  </p>
                </div>
                
                <div className={`p-4 rounded-lg border ${
                  isDark
                    ? 'bg-red-900 border-red-700'
                    : 'bg-red-50 border-red-200'
                }`}>
                  <p className="text-2xl font-bold text-red-600">{malignantCount}</p>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Malignant
                  </p>
                </div>
              </div>

              <div className={`p-4 rounded-lg border ${
                isDark
                  ? 'bg-blue-900 border-blue-700'
                  : 'bg-blue-50 border-blue-200'
              }`}>
                <p className="text-blue-600 font-medium">Average Ki-67 Index</p>
                <p className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  {avgKi67}%
                </p>
              </div>
            </div>
          )}

          {/* Individual Results with Visualization */}
          {processedFiles.length > 0 && (
            <div className="space-y-8 mt-8">
              <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Individual Analysis Results
              </h2>
              
              {processedFiles.map((batchFile, index) => (
                <div key={batchFile.id} className={`rounded-lg border-2 p-6 ${
                  isDark
                    ? 'bg-gray-800 border-gray-700'
                    : 'bg-white border-gray-200'
                }`}>
                  {/* File Header */}
                  <div className="mb-4">
                    <h3 className={`text-lg font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                      Case #{index + 1}: {batchFile.result?.batch_info?.original_filename || batchFile.file.name}
                    </h3>
                    <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                      Patient ID: {batchFile.result?.patient_data?.patient_id || `${patientPrefix}-${index + 1}`}
                    </p>
                  </div>

                  {batchFile.result && (
                    <div className="space-y-6">
                      {/* 4-Panel Visualization */}
                      <ImageVisualization
                        original={batchFile.result.images?.original}
                        analyzed={batchFile.result.images?.analyzed}
                        coordinates={batchFile.result.cell_coordinates}
                        ki67Index={batchFile.result.results?.ki67_index}
                      />

                      {/* Results Summary */}
                      <div>
                        <h4 className={`text-lg font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                          Results
                        </h4>
                        <ResultsCard
                          ki67Index={batchFile.result.results?.ki67_index || 0}
                          totalCells={batchFile.result.results?.total_cells || 0}
                          positiveCells={batchFile.result.results?.positive_cells || 0}
                          negativeCells={batchFile.result.results?.negative_cells || 0}
                        />
                      </div>

                      {/* Classification Badge */}
                      <div>
                        <ClassificationBadge
                          status={batchFile.result.results?.diagnosis?.classification || 'Unknown'}
                          risk={batchFile.result.results?.diagnosis?.risk || 'Unknown Risk'}
                        />
                      </div>

                      {/* Diagnosis Details */}
                      {batchFile.result.results?.diagnosis && (
                        <div className={`p-4 rounded-lg border ${
                          isDark
                            ? 'bg-gray-700 border-gray-600'
                            : 'bg-gray-50 border-gray-200'
                        }`}>
                          <h5 className={`font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                            Clinical Interpretation
                          </h5>
                          <p className={`text-sm mb-3 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                            {batchFile.result.results.diagnosis.interpretation}
                          </p>
                          <h5 className={`font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                            Recommendation
                          </h5>
                          <p className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                            {batchFile.result.results.diagnosis.recommendation}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}