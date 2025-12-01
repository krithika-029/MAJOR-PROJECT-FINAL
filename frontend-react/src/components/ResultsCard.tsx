import { useTheme } from '../contexts/ThemeContext';

interface MetricRowProps {
  label: string;
  value: string | number;
  color?: string;
}

function MetricRow({ label, value, color }: MetricRowProps) {
  const { isDark } = useTheme();
  const textColor = color === 'green' ? 'text-green-600' : color === 'red' ? 'text-red-600' : isDark ? 'text-white' : 'text-gray-900';

  return (
    <div className={`flex justify-between py-3 border-b last:border-0 ${
      isDark ? 'border-gray-700' : 'border-gray-200'
    }`}>
      <span className={`font-medium ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
        {label}
      </span>
      <span className={`font-bold ${textColor}`}>{value}</span>
    </div>
  );
}

interface ResultsCardProps {
  ki67Index: number;
  totalCells: number;
  positiveCells: number;
  negativeCells: number;
  manualPositive?: number;
  manualNegative?: number;
}

export default function ResultsCard({
  ki67Index,
  totalCells,
  positiveCells,
  negativeCells,
  manualPositive,
  manualNegative
}: ResultsCardProps) {
  const { isDark } = useTheme();

  return (
    <div className={`rounded-lg border-2 p-6 ${
      isDark
        ? 'bg-gray-800 border-gray-700'
        : 'bg-white border-gray-200'
    }`}>
      <h3 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
        Analysis Results
      </h3>
      <MetricRow label="Ki-67 Index" value={`${ki67Index.toFixed(1)}%`} />
      <MetricRow label="Total Cells" value={totalCells} />
      <MetricRow label="Positive Cells" value={`${positiveCells} (${((positiveCells/totalCells)*100).toFixed(0)}%)`} color="green" />
      <MetricRow label="Negative Cells" value={`${negativeCells} (${((negativeCells/totalCells)*100).toFixed(0)}%)`} color="red" />

      {manualPositive !== undefined && manualNegative !== undefined && (
        <div className="mt-6">
          <h4 className={`font-bold mb-3 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
            AI vs Manual Count
          </h4>
          <table className="w-full text-sm">
            <thead>
              <tr className={`border-b ${isDark ? 'border-gray-700' : 'border-gray-300'}`}>
                <th className={`text-left py-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                  Type
                </th>
                <th className={`text-right py-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                  AI Count
                </th>
                <th className={`text-right py-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                  Manual
                </th>
                <th className={`text-right py-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                  Diff
                </th>
              </tr>
            </thead>
            <tbody>
              <tr className={`border-b ${isDark ? 'border-gray-700' : 'border-gray-200'}`}>
                <td className={`py-2 ${isDark ? 'text-gray-300' : 'text-gray-900'}`}>
                  Positive
                </td>
                <td className={`text-right ${isDark ? 'text-gray-300' : 'text-gray-900'}`}>
                  {positiveCells}
                </td>
                <td className={`text-right ${isDark ? 'text-gray-300' : 'text-gray-900'}`}>
                  {manualPositive}
                </td>
                <td className={`text-right ${positiveCells >= manualPositive ? 'text-green-600' : 'text-red-600'}`}>
                  {positiveCells >= manualPositive ? '+' : ''}{positiveCells - manualPositive}
                </td>
              </tr>
              <tr>
                <td className={`py-2 ${isDark ? 'text-gray-300' : 'text-gray-900'}`}>
                  Negative
                </td>
                <td className={`text-right ${isDark ? 'text-gray-300' : 'text-gray-900'}`}>
                  {negativeCells}
                </td>
                <td className={`text-right ${isDark ? 'text-gray-300' : 'text-gray-900'}`}>
                  {manualNegative}
                </td>
                <td className={`text-right ${negativeCells >= manualNegative ? 'text-green-600' : 'text-red-600'}`}>
                  {negativeCells >= manualNegative ? '+' : ''}{negativeCells - manualNegative}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
