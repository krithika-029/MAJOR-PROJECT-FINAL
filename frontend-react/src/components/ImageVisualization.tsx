import { useTheme } from '../contexts/ThemeContext';
import { useState, useEffect } from 'react';

interface CellCoordinates {
  positive: [number, number][];
  negative: [number, number][];
}

interface ImageVisualizationProps {
  original?: string;
  analyzed?: string;
  coordinates?: CellCoordinates;
  ki67Index?: number;
}

export default function ImageVisualization({ original, analyzed, coordinates, ki67Index }: ImageVisualizationProps) {
  const { isDark } = useTheme();
  const [positiveImage, setPositiveImage] = useState<string | null>(null);
  const [negativeImage, setNegativeImage] = useState<string | null>(null);

  console.log('ImageVisualization rendering:', { 
    hasOriginal: !!original, 
    hasAnalyzed: !!analyzed,
    hasCoordinates: !!coordinates
  });

  if (!original && !analyzed) {
    return (
      <div className={`rounded-lg border-2 p-6 ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <h3 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Cell Detection Visualization
        </h3>
        <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          No visualization data available. Images will appear here after analysis.
        </p>
      </div>
    );
  }

  // Generate image with only positive markers
  const generatePositiveOnlyImage = () => {
    if (!original || !coordinates) return Promise.resolve(null);
    
    return new Promise<string | null>((resolve) => {
      const canvas = document.createElement('canvas');
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = original;
      
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return resolve(null);
        
        ctx.drawImage(img, 0, 0);
        
        const scaleX = img.width / 640;
        const scaleY = img.height / 640;
        
        // Draw only positive cells (green)
        coordinates.positive.forEach(([x, y]) => {
          const scaledX = x * scaleX;
          const scaledY = y * scaleY;
          
          ctx.strokeStyle = 'rgb(0, 255, 0)';
          ctx.fillStyle = 'rgb(0, 255, 0)';
          ctx.lineWidth = 2;
          
          ctx.beginPath();
          ctx.arc(scaledX, scaledY, 8, 0, 2 * Math.PI);
          ctx.stroke();
          
          ctx.beginPath();
          ctx.arc(scaledX, scaledY, 2, 0, 2 * Math.PI);
          ctx.fill();
        });
        
        resolve(canvas.toDataURL('image/png'));
      };
      
      img.onerror = () => resolve(null);
    });
  };

  // Generate image with only negative markers
  const generateNegativeOnlyImage = () => {
    if (!original || !coordinates) return Promise.resolve(null);
    
    return new Promise<string | null>((resolve) => {
      const canvas = document.createElement('canvas');
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = original;
      
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return resolve(null);
        
        ctx.drawImage(img, 0, 0);
        
        const scaleX = img.width / 640;
        const scaleY = img.height / 640;
        
        // Draw only negative cells (red)
        coordinates.negative.forEach(([x, y]) => {
          const scaledX = x * scaleX;
          const scaledY = y * scaleY;
          
          ctx.strokeStyle = 'rgb(255, 0, 0)';
          ctx.fillStyle = 'rgb(255, 0, 0)';
          ctx.lineWidth = 2;
          
          ctx.beginPath();
          ctx.arc(scaledX, scaledY, 8, 0, 2 * Math.PI);
          ctx.stroke();
          
          ctx.beginPath();
          ctx.arc(scaledX, scaledY, 2, 0, 2 * Math.PI);
          ctx.fill();
        });
        
        resolve(canvas.toDataURL('image/png'));
      };
      
      img.onerror = () => resolve(null);
    });
  };

  useEffect(() => {
    if (original && coordinates) {
      console.log('Generating separate visualizations...');
      generatePositiveOnlyImage().then(img => {
        if (img) setPositiveImage(img);
      });
      generateNegativeOnlyImage().then(img => {
        if (img) setNegativeImage(img);
      });
    }
  }, [original, coordinates]);

  return (
    <div className={`rounded-lg border-2 p-6 ${
      isDark
        ? 'bg-gray-800 border-gray-700'
        : 'bg-white border-gray-200'
    }`}>
      <h3 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
        Cell Detection Visualization
      </h3>
      
      {/* 2x2 Grid Layout */}
      <div className="grid grid-cols-2 gap-3">
        {/* Original Image */}
        <div className="relative">
          <div className={`absolute top-2 left-2 px-3 py-1 rounded text-sm font-semibold z-10 ${
            isDark ? 'bg-gray-900/80 text-white' : 'bg-white/90 text-gray-900'
          }`}>
            Original Image
          </div>
          {original && (
            <img 
              src={original} 
              alt="Original" 
              className="w-full rounded border border-gray-300"
            />
          )}
        </div>

        {/* Ki-67 Positive Only */}
        <div className="relative">
          <div className="absolute top-2 left-2 px-3 py-1 rounded text-sm font-semibold bg-green-600 text-white z-10">
            Ki-67 Positive: {coordinates?.positive.length || 0}
          </div>
          {positiveImage ? (
            <img 
              src={positiveImage} 
              alt="Ki-67 Positive" 
              className="w-full rounded border border-gray-300"
            />
          ) : original && (
            <img 
              src={original} 
              alt="Loading..." 
              className="w-full rounded border border-gray-300 opacity-50"
            />
          )}
        </div>

        {/* Ki-67 Negative Only */}
        <div className="relative">
          <div className="absolute top-2 left-2 px-3 py-1 rounded text-sm font-semibold bg-red-600 text-white z-10">
            Ki-67 Negative: {coordinates?.negative.length || 0}
          </div>
          {negativeImage ? (
            <img 
              src={negativeImage} 
              alt="Ki-67 Negative" 
              className="w-full rounded border border-gray-300"
            />
          ) : original && (
            <img 
              src={original} 
              alt="Loading..." 
              className="w-full rounded border border-gray-300 opacity-50"
            />
          )}
        </div>

        {/* Combined View */}
        <div className="relative">
          <div className="absolute top-2 left-2 px-3 py-1 rounded text-sm font-semibold bg-purple-600 text-white z-10">
            Combined (Ki-67: {ki67Index?.toFixed(1)}%)
          </div>
          {analyzed && (
            <img 
              src={analyzed} 
              alt="Combined Detection" 
              className="w-full rounded border border-gray-300"
            />
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-green-500 border-2 border-green-600"></div>
          <span className={isDark ? 'text-gray-300' : 'text-gray-700'}>Ki-67 Positive</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-red-500 border-2 border-red-600"></div>
          <span className={isDark ? 'text-gray-300' : 'text-gray-700'}>Ki-67 Negative</span>
        </div>
      </div>
    </div>
  );
}
