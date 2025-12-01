import { Upload } from 'lucide-react';
import { useState } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  multiple?: boolean;
}

export default function FileUpload({ onFileSelect, multiple = false }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-gray-50'
      }`}
    >
      <Upload className="mx-auto mb-4 text-gray-400" size={48} />
      <p className="text-gray-700 font-medium mb-2">
        Drag and drop an image here, or click to browse
      </p>
      <input
        type="file"
        onChange={handleChange}
        accept="image/*"
        multiple={multiple}
        className="hidden"
        id="file-upload"
      />
      <label
        htmlFor="file-upload"
        className="inline-block px-4 py-2 bg-blue-500 text-white rounded-lg cursor-pointer hover:bg-blue-600 transition-colors"
      >
        Choose File
      </label>
    </div>
  );
}
