import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, type TooltipProps,
} from 'recharts';
import { useTheme, themeColors } from './useTheme';

interface BarConfig {
  dataKey: string;
  name: string;
  color: string;
}

interface GroupedBarChartProps {
  data: Array<Record<string, string | number>>;
  xKey: string;
  bars: BarConfig[];
  yLabel?: string;
  yDomain?: [number, number];
  yTickFormatter?: (value: number) => string;
  height?: number;
  highlightFirst?: boolean;
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

export default function GroupedBarChart({
  data, xKey, bars, yLabel, yDomain, yTickFormatter, height = 300, highlightFirst = false,
}: GroupedBarChartProps) {
  const isDark = useTheme();
  const colors = themeColors(isDark);
  const fmt = yTickFormatter || ((v: number) => `${v}%`);

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
          <XAxis
            dataKey={xKey}
            tick={{ fill: colors.text, fontSize: 12 }}
            axisLine={{ stroke: colors.grid }}
            tickLine={false}
          />
          <YAxis
            domain={yDomain}
            tickFormatter={fmt}
            tick={{ fill: colors.text, fontSize: 12 }}
            axisLine={{ stroke: colors.grid }}
            tickLine={false}
            label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: colors.text, fontSize: 12 } : undefined}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '0.8rem', color: colors.text }}
          />
          {bars.map((bar) => (
            <Bar
              key={bar.dataKey}
              dataKey={bar.dataKey}
              name={bar.name}
              fill={bar.color}
              radius={[4, 4, 0, 0]}
              opacity={0.85}
            >
              {highlightFirst && data.map((_, i) => (
                <Cell key={i} opacity={i === 0 ? 1 : 0.6} />
              ))}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
