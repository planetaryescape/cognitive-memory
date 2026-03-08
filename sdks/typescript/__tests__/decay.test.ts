import { calculateRetention, updateStability } from "../src/core/decay";

describe("decay", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-02-10T00:00:00.000Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("procedural never decays", () => {
    const retention = calculateRetention({
      stability: 0.3,
      importance: 0.5,
      lastAccessed: Date.now() - 10_000_000_000,
      category: "procedural",
    });
    expect(retention).toBe(1.0);
  });

  test("fresh episodic ~0.98 at 1 day (stability 0.5, importance 0.5)", () => {
    // With episodic=45, S=0.5, B=2.0: effectiveRate=45, retention=exp(-1/45)=0.978
    const retention = calculateRetention({
      stability: 0.5,
      importance: 0.5,
      lastAccessed: Date.now() - 24 * 60 * 60 * 1000,
      category: "episodic",
    });
    expect(retention).toBeGreaterThan(0.97);
    expect(retention).toBeLessThan(0.99);
  });

  test("month-old episodic ~0.51 at 30 days (stability 0.5, importance 0.5)", () => {
    // With episodic=45, S=0.5, B=2.0: effectiveRate=45, retention=exp(-30/45)=0.513
    const retention = calculateRetention({
      stability: 0.5,
      importance: 0.5,
      lastAccessed: Date.now() - 30 * 24 * 60 * 60 * 1000,
      category: "episodic",
    });
    expect(retention).toBeGreaterThan(0.50);
    expect(retention).toBeLessThan(0.53);
  });

  test("higher importance slows decay", () => {
    const low = calculateRetention({
      stability: 0.5,
      importance: 0.1,
      lastAccessed: Date.now() - 30 * 24 * 60 * 60 * 1000,
      category: "semantic",
    });
    const high = calculateRetention({
      stability: 0.5,
      importance: 0.9,
      lastAccessed: Date.now() - 30 * 24 * 60 * 60 * 1000,
      category: "semantic",
    });
    expect(high).toBeGreaterThan(low);
  });

  test("higher stability slows decay", () => {
    const low = calculateRetention({
      stability: 0.2,
      importance: 0.5,
      lastAccessed: Date.now() - 30 * 24 * 60 * 60 * 1000,
      category: "semantic",
    });
    const high = calculateRetention({
      stability: 0.8,
      importance: 0.5,
      lastAccessed: Date.now() - 30 * 24 * 60 * 60 * 1000,
      category: "semantic",
    });
    expect(high).toBeGreaterThan(low);
  });

  test("core memories have 0.60 retention floor", () => {
    const retention = calculateRetention({
      stability: 0.1,
      importance: 0.1,
      lastAccessed: Date.now() - 365 * 24 * 60 * 60 * 1000,
      category: "core",
    });
    expect(retention).toBeGreaterThanOrEqual(0.60);
  });

  test("regular memories have 0.02 retention floor", () => {
    const retention = calculateRetention({
      stability: 0.1,
      importance: 0.1,
      lastAccessed: Date.now() - 365 * 24 * 60 * 60 * 1000,
      category: "episodic",
    });
    expect(retention).toBeGreaterThanOrEqual(0.02);
  });

  test("updateStability increases correctly + caps at 1.0", () => {
    expect(updateStability(0.3, 1)).toBeCloseTo(0.314, 3);
    expect(updateStability(0.3, 7)).toBeCloseTo(0.4, 6);
    expect(updateStability(0.3, 14)).toBeCloseTo(0.5, 6);
    expect(updateStability(0.95, 7)).toBe(1.0);
  });

  test("edge cases: negative days clamps to 0", () => {
    expect(updateStability(0.3, -10)).toBe(0.3);
    const retention = calculateRetention({
      stability: 0.5,
      importance: 0.5,
      lastAccessed: Date.now() + 10 * 24 * 60 * 60 * 1000,
      category: "semantic",
    });
    expect(retention).toBe(1.0);
  });

  test("backward compat: memoryType field still works", () => {
    const retention = calculateRetention({
      stability: 0.5,
      importance: 0.5,
      lastAccessed: Date.now() - 10_000_000_000,
      category: "procedural",
      memoryType: "procedural",
    });
    expect(retention).toBe(1.0);
  });
});
