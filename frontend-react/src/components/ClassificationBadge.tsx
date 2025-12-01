interface ClassificationBadgeProps {
  status: 'Benign' | 'Intermediate' | 'Malignant';
  risk: 'Low' | 'Medium' | 'High';
}

export default function ClassificationBadge({ status, risk }: ClassificationBadgeProps) {
  const colors = {
    Benign: 'bg-green-100 text-green-800 border-green-300',
    Intermediate: 'bg-yellow-100 text-yellow-800 border-yellow-300',
    Malignant: 'bg-red-100 text-red-800 border-red-300',
  };

  return (
    <div className={`inline-flex items-center px-6 py-3 rounded-lg border-2 ${colors[status]}`}>
      <div>
        <p className="text-sm font-medium">Classification</p>
        <p className="text-2xl font-bold">{status}</p>
        <p className="text-sm">Risk Level: {risk}</p>
      </div>
    </div>
  );
}
