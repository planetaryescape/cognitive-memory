import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, type TooltipProps,
} from 'recharts';
import { useTheme, themeColors } from './useTheme';

interface LineConfig {
  dataKey: string;
  name: string;
  color: string;
  dashed?: boolean;
}

interface ProgressionLineChartProps {
  data: Array<Record<string, string | number>>;
  xKey: string;
  lines: LineConfig[];
  yLabel?: string;
  yDomain?: [number, number];
  height?: number;
}

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  const isDark = useTheme();
  const colors = themeColors(isDark);
  if (!active || !payload?.length) return null;

  return (
    <div style={{
      background: colors.tooltip.bg,
      border: `1px solid ${colors.tooltip.border}`,
      borderRadius: '0.5rem',
      padding: '0.5rem 0.75rem',
      fontSize: '0.8rem',
    }}>
      <p style={{ margin: 0, fontWeight: 600, color: colors.tooltip.text }}>{label}</p>
      {payload.map((entry) => (
        <p key={entry.name} style={{ margin: '0.2rem 0 0', color: entry.color }}>
          {entry.name}: {typeof entry.value === 'number' ? `${entry.value}%` : entry.value}
        </p>
      ))}
    </div>
  );
}

export default function ProgressionLineChart({
  data, xKey, lines, yLabel, yDomain, height = 300,
}: ProgressionLineChartProps) {
  const isDark = useTheme();
  const colors = themeColors(isDark);

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
          <XAxis
            dataKey={xKey}
            tick={{ fill: colors.text, fontSize: 12 }}
            axisLine={{ stroke: colors.grid }}
            tickLine={false}
          />
          <YAxis
            domain={yDomain || [0, 65]}
            tickFormatter={(v: number) => `${v}%`}
            tick={{ fill: colors.text, fontSize: 12 }}
            axisLine={{ stroke: colors.grid }}
            tickLine={false}
            label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: colors.text, fontSize: 12 } : undefined}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: '0.8rem', color: colors.text }} />
          {lines.map((line) => (
            <Line
              key={line.dataKey}
              type="monotone"
              dataKey={line.dataKey}
              name={line.name}
              stroke={line.color}
              strokeWidth={2.5}
              strokeDasharray={line.dashed ? '5 5' : undefined}
              dot={{ r: 4, fill: line.color }}
              activeDot={{ r: 6 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
