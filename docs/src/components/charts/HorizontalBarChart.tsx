import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, type TooltipProps,
} from 'recharts';
import { useTheme, themeColors } from './useTheme';

interface HorizontalBarChartProps {
  data: Array<Record<string, string | number>>;
  yKey: string;
  dataKey: string;
  name?: string;
  color?: string;
  colors?: string[];
  xLabel?: string;
  xDomain?: [number, number];
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

export default function HorizontalBarChart({
  data, yKey, dataKey, name = 'Accuracy', color = '#8b5cf6',
  colors: colorArray, xLabel, xDomain, height = 250,
}: HorizontalBarChartProps) {
  const isDark = useTheme();
  const tc = themeColors(isDark);

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={tc.grid} horizontal={false} />
          <XAxis
            type="number"
            domain={xDomain || [0, 65]}
            tickFormatter={(v: number) => `${v}%`}
            tick={{ fill: tc.text, fontSize: 12 }}
            axisLine={{ stroke: tc.grid }}
            tickLine={false}
            label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -5, fill: tc.text, fontSize: 12 } : undefined}
          />
          <YAxis
            type="category"
            dataKey={yKey}
            tick={{ fill: tc.text, fontSize: 12 }}
            axisLine={{ stroke: tc.grid }}
            tickLine={false}
            width={75}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey={dataKey} name={name} fill={color} radius={[0, 4, 4, 0]} opacity={0.85}>
            {colorArray && data.map((_, i) => (
              <Cell key={i} fill={colorArray[i % colorArray.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
