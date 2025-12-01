import { useRef } from 'react';
import jsPDF from 'jspdf';
import { Download } from 'lucide-react';
import { Analysis } from '../utils/storage';
import { useTheme } from '../contexts/ThemeContext';

interface ReportGeneratorProps {
  analysis: Analysis;
  analysisResult?: any; // Full analysis result from backend with images (deprecated - now in analysis.analysisResult)
}

export default function ReportGenerator({ analysis }: ReportGeneratorProps) {
  const reportRef = useRef<HTMLDivElement>(null);
  const { isDark } = useTheme();
  
  // Get analysisResult from the analysis object (added by handleSave)
  const analysisResult = (analysis as any).analysisResult;

  const generatePDF = async () => {
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    let yPosition = 20;

    // Title with colored background
    pdf.setFillColor(0, 77, 64);
    pdf.rect(0, 0, pageWidth, 40, 'F');
    pdf.setFontSize(24);
    pdf.setTextColor(255, 255, 255);
    pdf.text('Ki-67 Proliferation Assessment Report', pageWidth / 2, 20, { align: 'center' });

    pdf.setFontSize(10);
    pdf.setTextColor(200, 255, 200);
    pdf.text(`Generated: ${new Date().toLocaleString()}`, pageWidth / 2, 30, { align: 'center' });

    yPosition = 50;
    pdf.setFontSize(14);
    pdf.setTextColor(0, 77, 64);
    pdf.setFont(undefined, 'bold');
    pdf.text('PATIENT INFORMATION', 20, yPosition);

    yPosition += 8;
    pdf.setFont(undefined, 'normal');
    pdf.setFontSize(10);
    pdf.setTextColor(0, 0, 0);
    
    // Patient info box with light blue background
    pdf.setFillColor(224, 247, 250);
    pdf.rect(20, yPosition - 3, pageWidth - 40, 30, 'F');
    pdf.setDrawColor(0, 77, 64);
    pdf.rect(20, yPosition - 3, pageWidth - 40, 30, 'S');
    
    pdf.text(`Patient ID: ${analysis.patientId}`, 25, yPosition + 3);
    yPosition += 7;
    pdf.text(`Analysis Date: ${analysis.date}`, 25, yPosition + 3);
    yPosition += 7;
    pdf.text(`Analysis ID: ${analysis.id}`, 25, yPosition + 3);
    yPosition += 7;
    if (analysis.notes) {
      pdf.text(`Clinical Notes: ${analysis.notes}`, 25, yPosition + 3);
      yPosition += 7;
    }
    if (analysis.imageName) {
      pdf.text(`Image: ${analysis.imageName}`, 25, yPosition + 3);
    }

    yPosition += 15;
    
    // Add visualization image if available
    if (analysisResult?.images?.analyzed) {
      pdf.setFontSize(14);
      pdf.setFont(undefined, 'bold');
      pdf.setTextColor(0, 77, 64);
      pdf.text('CELL DETECTION VISUALIZATION', 20, yPosition);
      yPosition += 8;
      
      try {
        const imgWidth = 150;
        const imgHeight = 110;
        pdf.addImage(analysisResult.images.analyzed, 'PNG', (pageWidth - imgWidth) / 2, yPosition, imgWidth, imgHeight);
        yPosition += imgHeight + 5;
        
        pdf.setFontSize(8);
        pdf.setTextColor(100, 100, 100);
        pdf.text('Figure: Cell detection with Ki-67 positive cells (red circles) and negative cells (blue circles)', pageWidth / 2, yPosition, { align: 'center' });
        yPosition += 10;
      } catch (e) {
        console.error('Could not add image to PDF:', e);
      }
    }

    yPosition += 5;
    pdf.setFontSize(14);
    pdf.setFont(undefined, 'bold');
    pdf.setTextColor(0, 77, 64);
    pdf.text('ANALYSIS RESULTS SUMMARY', 20, yPosition);

    yPosition += 8;
    pdf.setFont(undefined, 'normal');
    pdf.setFontSize(10);

    // Determine background color based on risk
    let bgColor = [232, 245, 233]; // green for benign
    let headerColor = [76, 175, 80];
    if (analysis.status === 'Malignant') {
      bgColor = [255, 235, 238];
      headerColor = [244, 67, 54];
    } else if (analysis.status === 'Borderline Malignant' || analysis.status === 'Low Malignant Potential') {
      bgColor = [255, 243, 224];
      headerColor = [255, 152, 0];
    }

    const resultData = [
      [`Metric`, `Value`],
      [`Ki-67 Proliferation Index`, `${analysis.ki67Index.toFixed(1)}%`],
      [`Total Cells Detected`, `${analysis.totalCells}`],
      [`Positive Cells (Ki-67+)`, `${analysis.positiveCells} (${((analysis.positiveCells / analysis.totalCells) * 100).toFixed(1)}%)`],
      [`Negative Cells (Ki-67-)`, `${analysis.negativeCells} (${((analysis.negativeCells / analysis.totalCells) * 100).toFixed(1)}%)`],
      [`Diagnosis Classification`, `${analysis.status}`],
      [`Risk Level`, `${analysis.risk}`],
      [`Malignancy Status`, analysis.status === 'Malignant' ? 'Malignant' : 'Non-Malignant'],
    ];

    resultData.forEach((row, index) => {
      if (index === 0) {
        pdf.setFont(undefined, 'bold');
        pdf.setFillColor(headerColor[0], headerColor[1], headerColor[2]);
        pdf.setTextColor(255, 255, 255);
        pdf.rect(20, yPosition - 5, pageWidth - 40, 8, 'F');
      } else {
        pdf.setFont(undefined, 'normal');
        pdf.setFillColor(bgColor[0], bgColor[1], bgColor[2]);
        pdf.setTextColor(0, 0, 0);
        pdf.rect(20, yPosition - 5, pageWidth - 40, 8, 'FD');
      }
      pdf.text(row[0], 25, yPosition);
      pdf.text(row[1], pageWidth - 25, yPosition, { align: 'right' });
      yPosition += 8;
    });

    yPosition += 10;
    pdf.setFontSize(14);
    pdf.setFont(undefined, 'bold');
    pdf.setTextColor(0, 77, 64);
    pdf.text('CLINICAL INTERPRETATION', 20, yPosition);

    yPosition += 8;
    pdf.setFont(undefined, 'normal');
    pdf.setFontSize(10);
    pdf.setTextColor(0, 0, 0);

    // Get interpretation from backend or use default
    let interpretation = '';
    if (analysisResult?.results?.diagnosis?.interpretation) {
      interpretation = analysisResult.results.diagnosis.interpretation;
    } else {
      const interpretations: { [key: string]: string } = {
        'Benign': 'Low Ki-67 index (<5%) indicates very low proliferation rate. Consistent with benign characteristics. Low risk of malignancy.',
        'Low Malignant Potential': 'Low to moderate Ki-67 index (5-10%) suggests low proliferation. May require monitoring and follow-up evaluation.',
        'Borderline Malignant': 'Moderate Ki-67 index (10-20%) indicates moderate proliferation rate. Borderline malignant characteristics. Further clinical evaluation strongly recommended.',
        'Malignant': 'High Ki-67 index (>20%) indicates high proliferation rate. Consistent with malignant tumor characteristics. Requires immediate clinical intervention.',
      };
      interpretation = interpretations[analysis.status] || 'Clinical evaluation recommended.';
    }

    // Word wrap for interpretation with better width calculation
    const maxWidth = pageWidth - 45; // Reduced from 50 for more margin
    const leftMargin = 25;
    const rightMargin = pageWidth - 20;
    const lineHeight = 5;

    // Helper function for proper text wrapping
    const wrapText = (text: string, startY: number) => {
      let currentY = startY;
      const words = text.split(' ');
      let line = '';
      
      words.forEach((word, index) => {
        const testLine = line + word + ' ';
        const testWidth = pdf.getTextWidth(testLine);
        
        if (testWidth > maxWidth && line.length > 0) {
          // Check if need new page
          if (currentY > pageHeight - 30) {
            pdf.addPage();
            currentY = 20;
          }
          pdf.text(line.trim(), leftMargin, currentY);
          currentY += lineHeight;
          line = word + ' ';
        } else {
          line = testLine;
        }
      });
      
      // Print remaining text
      if (line.trim().length > 0) {
        if (currentY > pageHeight - 30) {
          pdf.addPage();
          currentY = 20;
        }
        pdf.text(line.trim(), leftMargin, currentY);
        currentY += lineHeight;
      }
      
      return currentY;
    };

    yPosition = wrapText(interpretation, yPosition);

    yPosition += 8;
    
    // Check if need new page before recommendation
    if (yPosition > pageHeight - 50) {
      pdf.addPage();
      yPosition = 20;
    }
    
    pdf.setFontSize(14);
    pdf.setFont(undefined, 'bold');
    pdf.setTextColor(216, 67, 21);
    pdf.text('CLINICAL RECOMMENDATION', 20, yPosition);

    yPosition += 8;
    pdf.setFont(undefined, 'normal');
    pdf.setFontSize(10);
    pdf.setTextColor(0, 0, 0);

    let recommendation = '';
    if (analysisResult?.results?.diagnosis?.recommendation) {
      recommendation = analysisResult.results.diagnosis.recommendation;
    } else {
      const recommendations: { [key: string]: string } = {
        'Benign': 'Routine follow-up. No immediate intervention required. Continue standard monitoring protocol.',
        'Low Malignant Potential': 'Clinical monitoring recommended. Follow-up imaging suggested within 3-6 months.',
        'Borderline Malignant': 'Further diagnostic evaluation recommended. Consider additional histopathological studies and imaging. Multidisciplinary team review advised.',
        'Malignant': 'Immediate oncological consultation required. Comprehensive treatment planning needed. Additional staging studies recommended.',
      };
      recommendation = recommendations[analysis.status] || 'Consult with oncologist for appropriate management.';
    }

    // Use the same wrap function for recommendation
    yPosition = wrapText(recommendation, yPosition);

    // Reference information
    yPosition += 10;
    
    // Check if need new page before reference
    if (yPosition > pageHeight - 60) {
      pdf.addPage();
      yPosition = 20;
    }
    
    pdf.setFontSize(12);
    pdf.setFont(undefined, 'bold');
    pdf.setTextColor(0, 77, 64);
    pdf.text('REFERENCE GUIDELINES', 20, yPosition);

    yPosition += 7;
    pdf.setFontSize(9);
    pdf.setFont(undefined, 'normal');
    pdf.setTextColor(60, 60, 60);
    
    const refLines = [
      'Ki-67 Proliferation Index Interpretation:',
      '  • <5%: Very Low Proliferation (Benign)',
      '  • 5-10%: Low Proliferation (Low Malignant Potential)',
      '  • 10-20%: Moderate Proliferation (Borderline Malignant)',
      '  • >20%: High Proliferation (Malignant)',
      '',
      'Note: This report is generated by an AI-assisted analysis system.',
      'Results should be reviewed by a qualified pathologist and correlated',
      'with clinical findings.',
    ];

    refLines.forEach(refLine => {
      // Check if need new page
      if (yPosition > pageHeight - 30) {
        pdf.addPage();
        yPosition = 20;
      }
      // Use proper margin
      pdf.text(refLine, leftMargin - 3, yPosition);
      yPosition += 4;
    });

    // Footer
    yPosition = pageHeight - 15;
    pdf.setFillColor(0, 77, 64);
    pdf.rect(0, yPosition - 5, pageWidth, 20, 'F');
    pdf.setFontSize(8);
    pdf.setTextColor(255, 255, 255);
    pdf.text(`Report Generated: ${new Date().toLocaleString()} | Analysis ID: ${analysis.id}`, pageWidth / 2, yPosition, { align: 'center' });
    pdf.text('Ki-67 Medical Diagnostic System | Confidential Medical Report', pageWidth / 2, yPosition + 5, { align: 'center' });

    pdf.save(`Ki67_Report_${analysis.patientId}_${new Date().toISOString().split('T')[0]}.pdf`);
  };

  return (
    <button
      onClick={generatePDF}
      className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-colors bg-green-500 hover:bg-green-600 text-white"
    >
      <Download size={18} />
      Download PDF Report
    </button>
  );
}
