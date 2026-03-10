export function assertNonEmptyString(field: string, value: string) {
  if (value.trim().length === 0) {
    throw new Error(`Invalid ${field}: ${value} (must be non-empty string)`);
  }
}

export function assertUnitInterval(field: string, value: number) {
  if (Number.isNaN(value) || value < 0 || value > 1) {
    throw new Error(`Invalid ${field}: ${value} (must be [0.0, 1.0])`);
  }
}

export function assertNonNegativeInt(field: string, value: number) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(
      `Invalid ${field}: ${value} (must be non-negative integer)`,
    );
  }
}
