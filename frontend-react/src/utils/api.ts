// API utilities for Ki-67 Analysis System
const API_BASE_URL = '/api';  // Use proxy during development, direct path in production

export interface AnalysisRequest {
  image: File;
  patientId: string;
  manualPositive?: number;
  manualNegative?: number;
  notes?: string;
}

export interface BatchAnalysisRequest {
  images: File[];
  patientPrefix?: string;
}

export interface AnalysisResult {
  analysis_id: string;
  timestamp: string;
  patient_data?: {
    patient_id?: string;
  };
  results: {
    positive_cells: number;
    negative_cells: number;
    total_cells: number;
    ki67_index: number;
    diagnosis: string;
  };
  images: {
    original: string;
    analyzed: string;
    ground_truth?: string;
    comparison_overlay?: string;
  };
  cell_coordinates: {
    positive: [number, number][];
    negative: [number, number][];
    ground_truth_positive?: [number, number][];
    ground_truth_negative?: [number, number][];
  };
  // Legacy fields for compatibility
  id?: string;
  ki67Index?: number;
  totalCells?: number;
  positiveCells?: number;
  negativeCells?: number;
  coordinates?: {
    positive: [number, number][];
    negative: [number, number][];
  };
  classification?: {
    status: string;
    risk: string;
    description: string;
    recommendation: string;
  };
}

export interface BatchAnalysisResult {
  results: AnalysisResult[];
}

// Check if API is available
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    return data.status === 'healthy' && data.model_loaded;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
}

// Single image analysis
export async function analyzeImage(request: AnalysisRequest): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('image', request.image);
  formData.append('patientId', request.patientId);
  
  if (request.manualPositive !== undefined) {
    formData.append('manualPositive', request.manualPositive.toString());
  }
  if (request.manualNegative !== undefined) {
    formData.append('manualNegative', request.manualNegative.toString());
  }
  if (request.notes) {
    formData.append('notes', request.notes);
  }

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Analysis failed');
  }

  return response.json();
}

// Batch analysis
export async function analyzeBatch(request: BatchAnalysisRequest): Promise<BatchAnalysisResult> {
  const formData = new FormData();
  
  request.images.forEach((image) => {
    formData.append('images', image);
  });
  
  if (request.patientPrefix) {
    formData.append('patientPrefix', request.patientPrefix);
  }

  const response = await fetch(`${API_BASE_URL}/analyze/batch-upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Batch analysis failed');
  }

  return response.json();
}

// Utility to classify tumor (moved from storage.ts)
export function classifyTumor(ki67Index: number) {
  if (ki67Index < 14) {
    return {
      status: 'Low Grade',
      risk: 'Low Risk',
      description: 'Low proliferative activity',
      recommendation: 'Standard follow-up recommended'
    };
  } else if (ki67Index < 30) {
    return {
      status: 'Intermediate Grade', 
      risk: 'Moderate Risk',
      description: 'Moderate proliferative activity',
      recommendation: 'Close monitoring and targeted therapy consideration'
    };
  } else {
    return {
      status: 'High Grade',
      risk: 'High Risk', 
      description: 'High proliferative activity',
      recommendation: 'Aggressive treatment protocol recommended'
    };
  }
}