export interface Analysis {
  id: string;
  date: string;
  patientId: string;
  ki67Index: number;
  status: 'Benign' | 'Intermediate' | 'Malignant';
  risk: 'Low' | 'Medium' | 'High';
  positiveCells: number;
  negativeCells: number;
  totalCells: number;
  imageName?: string;
  notes?: string;
}

const STORAGE_KEY = 'ki67_analyses';

export function classifyTumor(ki67Index: number): { status: 'Benign' | 'Intermediate' | 'Malignant'; risk: 'Low' | 'Medium' | 'High'; color: string } {
  if (ki67Index < 5) return { status: 'Benign', risk: 'Low', color: 'green' };
  if (ki67Index < 30) return { status: 'Intermediate', risk: 'Medium', color: 'yellow' };
  return { status: 'Malignant', risk: 'High', color: 'red' };
}

export function initializeSampleData() {
  const existing = localStorage.getItem(STORAGE_KEY);
  if (!existing) {
    const sampleData: Analysis[] = [
      { id: '1', patientId: 'P001', ki67Index: 3.2, status: 'Benign', risk: 'Low', positiveCells: 16, negativeCells: 484, totalCells: 500, date: '2025-11-20', imageName: 'sample_01.jpg' },
      { id: '2', patientId: 'P002', ki67Index: 54.3, status: 'Malignant', risk: 'High', positiveCells: 102, negativeCells: 85, totalCells: 187, date: '2025-11-22', imageName: 'sample_02.jpg' },
      { id: '3', patientId: 'P003', ki67Index: 12.5, status: 'Intermediate', risk: 'Medium', positiveCells: 50, negativeCells: 350, totalCells: 400, date: '2025-11-23', imageName: 'sample_03.jpg' },
      { id: '4', patientId: 'P004', ki67Index: 2.8, status: 'Benign', risk: 'Low', positiveCells: 14, negativeCells: 486, totalCells: 500, date: '2025-11-24', imageName: 'sample_04.jpg' },
      { id: '5', patientId: 'P005', ki67Index: 45.2, status: 'Malignant', risk: 'High', positiveCells: 113, negativeCells: 137, totalCells: 250, date: '2025-11-24', imageName: 'sample_05.jpg' },
      { id: '6', patientId: 'P006', ki67Index: 8.3, status: 'Intermediate', risk: 'Medium', positiveCells: 25, negativeCells: 275, totalCells: 300, date: '2025-11-25', imageName: 'sample_06.jpg' },
      { id: '7', patientId: 'P007', ki67Index: 1.5, status: 'Benign', risk: 'Low', positiveCells: 6, negativeCells: 394, totalCells: 400, date: '2025-11-25', imageName: 'sample_07.jpg' },
      { id: '8', patientId: 'P008', ki67Index: 67.8, status: 'Malignant', risk: 'High', positiveCells: 203, negativeCells: 97, totalCells: 300, date: '2025-11-26', imageName: 'sample_08.jpg' },
      { id: '9', patientId: 'P009', ki67Index: 15.6, status: 'Intermediate', risk: 'Medium', positiveCells: 78, negativeCells: 422, totalCells: 500, date: '2025-11-26', imageName: 'sample_09.jpg' },
      { id: '10', patientId: 'P010', ki67Index: 4.2, status: 'Benign', risk: 'Low', positiveCells: 21, negativeCells: 479, totalCells: 500, date: '2025-11-27', imageName: 'sample_10.jpg' },
    ];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sampleData));
  }
}

export function getAnalyses(): Analysis[] {
  const data = localStorage.getItem(STORAGE_KEY);
  return data ? JSON.parse(data) : [];
}

const MAX_HISTORY_ITEMS = 10;

export function saveAnalysis(analysis: Analysis) {
  // Strip heavy fields (images, coordinates) to avoid localStorage quota exceeded
  const { images, cell_coordinates, ...minimal }: any = analysis;
  
  const analyses = getAnalyses();
  // Remove duplicate if exists
  const filtered = analyses.filter((a: any) => a.id !== minimal.id && a.analysis_id !== minimal.analysis_id);
  filtered.unshift(minimal);
  
  // Cap to MAX_HISTORY_ITEMS to prevent storage overflow
  const capped = filtered.slice(0, MAX_HISTORY_ITEMS);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(capped));
}

export function clearHistory() {
  localStorage.removeItem(STORAGE_KEY);
}

export function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}
