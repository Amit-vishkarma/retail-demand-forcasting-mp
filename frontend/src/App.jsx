import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  BarChart3,
  Boxes,
  CalendarDays,
  ChartNoAxesCombined,
  ChevronDown,
  Download,
  LayoutDashboard,
  LoaderCircle,
  PackageSearch,
  RefreshCw,
  ShoppingBasket,
  Store,
  TrendingUp,
} from "lucide-react";

const API = "https://retail-demand-forcasting-mp.onrender.com";
const months = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];
const money = (n) =>
  Number(n || 0).toLocaleString("en-IN", { maximumFractionDigits: 1 });

function StatCard({ icon: Icon, label, value, hint }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{label}</span>
        <span className="stat-icon">
          <Icon size={18} />
        </span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-hint">{hint}</div>
    </div>
  );
}
function Panel({ title, subtitle, children, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}
function EmptyState({ children }) {
  return <div className="empty-state">{children}</div>;
}

export default function App() {
  const [page, setPage] = useState("Overview");
  const [summary, setSummary] = useState(null);
  const [options, setOptions] = useState({ stores: [], categories: [] });
  const [evaluation, setEvaluation] = useState(null);
  const [store, setStore] = useState("");
  const [category, setCategory] = useState("");
  const [month, setMonth] = useState(10);
  const [year, setYear] = useState(2026);
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [error, setError] = useState("");

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      const [s, o, e] = await Promise.all([
        axios.get(`${API}/api/summary`),
        axios.get(`${API}/api/options`),
        axios.get(`${API}/api/evaluation`),
      ]);
      setSummary(s.data);
      setOptions(o.data);
      setEvaluation(e.data);
      setStore((current) => current || o.data.stores[0] || "");
      setCategory((current) => current || o.data.categories[0] || "");
    } catch (err) {
      setError(
        `Backend is not connected (${API}). ${err.response?.data?.detail || err.message}`,
      );
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    refresh();
  }, []);

  const forecastData = useMemo(
    () =>
      forecast?.daily?.map((x) => ({
        ...x,
        dateLabel: String(x.Date).slice(5),
      })) || [],
    [forecast],
  );
  const downloadForecast = () => {
    if (!forecast?.daily?.length) return;
    const rows = [
      ["Date", "Store ID", "Category", "Predicted Demand"],
      ...forecast.daily.map((r) => [
        r.Date,
        r["Store ID"],
        r.Category,
        r["Predicted Demand"],
      ]),
    ];
    const csv = rows
      .map((row) =>
        row.map((v) => `"${String(v ?? "").replaceAll('"', '""')}"`).join(","),
      )
      .join("\n");
    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8;" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `forecast_${store}_${category}_${year}_${month}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const generateForecast = async (e) => {
    e.preventDefault();
    setForecastLoading(true);
    setError("");
    setForecast(null);
    try {
      const response = await axios.post(`${API}/api/forecast`, {
        store_id: store,
        category,
        year: Number(year),
        month: Number(month),
      });
      setForecast(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Forecast failed.");
    } finally {
      setForecastLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ChartNoAxesCombined size={23} />
          </div>
          <div>
            <strong>DemandIQ</strong>
            <span>RETAIL ANALYTICS</span>
          </div>
        </div>
        <div className="nav-label">WORKSPACE</div>
        <button
          className={`nav-item ${page === "Overview" ? "active" : ""}`}
          onClick={() => setPage("Overview")}
        >
          <LayoutDashboard size={18} /> Overview
        </button>
        <button
          className={`nav-item ${page === "Forecast" ? "active" : ""}`}
          onClick={() => setPage("Forecast")}
        >
          <TrendingUp size={18} /> Demand forecast
        </button>
        <button
          className={`nav-item ${page === "Evaluation" ? "active" : ""}`}
          onClick={() => setPage("Evaluation")}
        >
          <BarChart3 size={18} /> Model evaluation
        </button>
        <div className="sidebar-bottom">
          <div className="status-dot" />{" "}
          <div>
            <b>ML service</b>
            <span>
              {loading
                ? "Connecting…"
                : error
                  ? "Connection issue"
                  : "Dashboard ready"}
            </span>
          </div>
          <button
            title="Refresh data"
            className="icon-button"
            onClick={refresh}
          >
            <RefreshCw size={16} />
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <b>{page}</b>
          </div>
          <div className="topbar-right">
            <span className="live-indicator">
              <i /> Live dashboard
            </span>
            <div className="avatar">RF</div>
          </div>
        </header>
        <div className="page-content">
          {error && (
            <div className="alert">
              <b>Notice</b>
              <span>{error}</span>
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
          {loading ? (
            <div className="loading-page">
              <LoaderCircle className="spin" size={30} />
              <span>Loading analytics…</span>
            </div>
          ) : (
            <>
              {page === "Overview" && (
                <>
                  <div className="page-title-row">
                    <div>
                      <div className="eyebrow">RETAIL INTELLIGENCE</div>
                      <h1>Overview</h1>
                      <p className="subtitle">
                        A clear view of demand across your stores and product
                        categories.
                      </p>
                    </div>
                    <button className="secondary-button" onClick={refresh}>
                      <RefreshCw size={16} /> Refresh data
                    </button>
                  </div>
                  <div className="date-banner">
                    <CalendarDays size={18} />
                    <span>Dataset period</span>
                    <b>
                      {summary?.date_start} — {summary?.date_end}
                    </b>
                    <span className="banner-pill">Updated dataset</span>
                  </div>
                  <div className="stats-grid">
                    <StatCard
                      icon={Activity}
                      label="Total demand"
                      value={money(summary?.total_demand)}
                      hint="Units across all records"
                    />
                    <StatCard
                      icon={Store}
                      label="Active stores"
                      value={money(summary?.stores)}
                      hint="Unique store IDs"
                    />
                    <StatCard
                      icon={Boxes}
                      label="Products"
                      value={money(summary?.products)}
                      hint="Unique product IDs"
                    />
                    <StatCard
                      icon={TrendingUp}
                      label="Average demand"
                      value={money(summary?.average_demand)}
                      hint="Units per record"
                    />
                  </div>
                  <div className="chart-grid overview-charts">
                    <Panel
                      title="Demand trend"
                      subtitle="Daily total observed demand"
                      className="wide-panel"
                    >
                      <div className="chart-wrap tall">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={summary?.daily || []}>
                            <defs>
                              <linearGradient
                                id="demandFill"
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                              >
                                <stop
                                  offset="0%"
                                  stopColor="#7165f2"
                                  stopOpacity={0.23}
                                />
                                <stop
                                  offset="95%"
                                  stopColor="#7165f2"
                                  stopOpacity={0.01}
                                />
                              </linearGradient>
                            </defs>
                            <CartesianGrid
                              strokeDasharray="3 3"
                              vertical={false}
                              stroke="#edf0f5"
                            />
                            <XAxis
                              dataKey="date"
                              tickFormatter={(v) => v.slice(5)}
                              minTickGap={28}
                            />
                            <YAxis />
                            <Tooltip />
                            <Area
                              type="monotone"
                              dataKey="demand"
                              stroke="#7165f2"
                              fill="url(#demandFill)"
                              strokeWidth={2}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </Panel>
                    <Panel
                      title="Demand by category"
                      subtitle="Total units grouped by category"
                    >
                      <div className="chart-wrap tall">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={summary?.categories || []}
                            layout="vertical"
                            margin={{ left: 8, right: 10 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              horizontal={false}
                              stroke="#edf0f5"
                            />
                            <XAxis type="number" />
                            <YAxis
                              type="category"
                              dataKey="category"
                              width={95}
                              tick={{ fontSize: 11 }}
                            />
                            <Tooltip />
                            <Bar
                              dataKey="demand"
                              fill="#7165f2"
                              radius={[0, 5, 5, 0]}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </Panel>
                  </div>
                  <Panel
                    title="Top products"
                    subtitle="Highest total observed demand"
                  >
                    <div className="chart-wrap medium">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={summary?.top_products || []}
                          margin={{ top: 10, right: 10, bottom: 35, left: 10 }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            vertical={false}
                            stroke="#edf0f5"
                          />

                          <XAxis
                            dataKey="product_name"
                            interval={0}
                            angle={-25}
                            textAnchor="end"
                            height={75}
                            tick={{ fontSize: 11 }}
                          />

                          <YAxis />

                          <Tooltip
                            formatter={(value) => [
                              money(value),
                              "Total demand",
                            ]}
                            labelFormatter={(label) => `Product: ${label}`}
                          />

                          <Bar
                            dataKey="demand"
                            name="Total demand"
                            fill="#25a887"
                            radius={[5, 5, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </Panel>
                </>
              )}

              {page === "Forecast" && (
                <>
                  <div className="page-title-row">
                    <div>
                      <div className="eyebrow">PREDICTIVE ANALYTICS</div>
                      <h1>Demand forecast</h1>
                      <p className="subtitle">
                        Choose a store, category and month to estimate future
                        demand.
                      </p>
                    </div>
                    <span className="model-badge">
                      <span /> Extra Trees Regressor
                    </span>
                  </div>
                  <div className="forecast-layout">
                    <Panel
                      title="Forecast settings"
                      subtitle="Select the period and segment to predict"
                    >
                      <form
                        className="forecast-form"
                        onSubmit={generateForecast}
                      >
                        <label>
                          Store
                          <select
                            value={store}
                            onChange={(e) => setStore(e.target.value)}
                            required
                          >
                            {options.stores.map((x) => (
                              <option key={x} value={x}>
                                {x}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Category
                          <select
                            value={category}
                            onChange={(e) => setCategory(e.target.value)}
                            required
                          >
                            {options.categories.map((x) => (
                              <option key={x} value={x}>
                                {x}
                              </option>
                            ))}
                          </select>
                        </label>
                        <div className="form-row">
                          <label>
                            Forecast month
                            <select
                              value={month}
                              onChange={(e) => setMonth(Number(e.target.value))}
                            >
                              {months.map((m, i) => (
                                <option key={m} value={i + 1}>
                                  {m}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label>
                            Year
                            <select
                              value={year}
                              onChange={(e) => setYear(Number(e.target.value))}
                            >
                              {[2026, 2027, 2028, 2029, 2030].map((y) => (
                                <option key={y}>{y}</option>
                              ))}
                            </select>
                          </label>
                        </div>
                        <div className="form-note">
                          <CalendarDays size={16} />
                          <span>
                            Forecast runs for every day in {months[month - 1]}{" "}
                            {year} and aggregates products in the selected
                            category.
                          </span>
                        </div>
                        <button
                          className="primary-button"
                          type="submit"
                          disabled={forecastLoading || !store || !category}
                        >
                          {forecastLoading ? (
                            <>
                              <LoaderCircle className="spin" size={17} />{" "}
                              Generating…
                            </>
                          ) : (
                            <>
                              <TrendingUp size={17} /> Generate forecast
                            </>
                          )}
                        </button>
                      </form>
                    </Panel>
                    <Panel
                      title="How this forecast works"
                      subtitle="Important assumptions"
                    >
                      <div className="assumption-list">
                        <div>
                          <span className="assumption-number">01</span>
                          <p>
                            <b>Historical demand</b>
                            <br />
                            Recent observed demand provides the lag features
                            used by the model.
                          </p>
                        </div>
                        <div>
                          <span className="assumption-number">02</span>
                          <p>
                            <b>Future business inputs</b>
                            <br />
                            Inventory, price, discount and promotion are held at
                            the latest available values.
                          </p>
                        </div>
                        <div>
                          <span className="assumption-number">03</span>
                          <p>
                            <b>Category-level output</b>
                            <br />
                            Predictions for eligible products are summed into a
                            daily category forecast.
                          </p>
                        </div>
                      </div>
                    </Panel>
                  </div>
                  {forecast && (
                    <>
                      <div className="forecast-result-heading">
                        <div>
                          <h2>{category} demand forecast</h2>
                          <p>
                            {months[month - 1]} {year} · Store {store} ·{" "}
                            {forecast.product_count} products
                          </p>
                        </div>
                        <button
                          className="secondary-button"
                          onClick={downloadForecast}
                        >
                          <Download size={16} /> Download CSV
                        </button>
                      </div>
                      <div className="stats-grid three-stats">
                        <StatCard
                          icon={TrendingUp}
                          label="Forecast total"
                          value={money(forecast.total_demand)}
                          hint="Estimated units this month"
                        />
                        <StatCard
                          icon={Activity}
                          label="Daily average"
                          value={money(forecast.average_daily_demand)}
                          hint="Average predicted units/day"
                        />
                        <StatCard
                          icon={PackageSearch}
                          label="Products forecast"
                          value={money(forecast.product_count)}
                          hint="Products with sufficient history"
                        />
                      </div>
                      <Panel
                        title="Daily forecast"
                        subtitle="Predicted demand for each day"
                      >
                        <div className="chart-wrap tall">
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={forecastData}>
                              <CartesianGrid
                                strokeDasharray="3 3"
                                vertical={false}
                                stroke="#edf0f5"
                              />
                              <XAxis dataKey="dateLabel" />
                              <YAxis />
                              <Tooltip />
                              <Area
                                type="monotone"
                                dataKey="Predicted Demand"
                                stroke="#7165f2"
                                fill="#7165f2"
                                fillOpacity={0.12}
                                strokeWidth={2}
                              />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      </Panel>
                      <Panel
                        title="Forecast data"
                        subtitle="Daily category totals"
                      >
                        <div className="table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>Date</th>
                                <th>Store</th>
                                <th>Category</th>
                                <th>Predicted demand</th>
                              </tr>
                            </thead>
                            <tbody>
                              {forecast.daily.map((row) => (
                                <tr key={row.Date}>
                                  <td>{row.Date}</td>
                                  <td>{row["Store ID"]}</td>
                                  <td>{row.Category}</td>
                                  <td>
                                    <b>{money(row["Predicted Demand"])}</b>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        <p className="footnote">{forecast.assumption}</p>
                      </Panel>
                    </>
                  )}
                </>
              )}

              {page === "Evaluation" && (
                <>
                  <div className="page-title-row">
                    <div>
                      <div className="eyebrow">MODEL PERFORMANCE</div>
                      <h1>Model evaluation</h1>
                      <p className="subtitle">
                        Chronological holdout evaluation of the trained Extra
                        Trees model.
                      </p>
                    </div>
                    <span className="model-badge">
                      <span /> Extra Trees Regressor
                    </span>
                  </div>
                  {!evaluation?.available ? (
                    <EmptyState>
                      {evaluation?.message ||
                        "Evaluation metrics not available. Run model training first."}
                    </EmptyState>
                  ) : (
                    <>
                      <div className="date-banner">
                        <CalendarDays size={18} />
                        <span>Test period</span>
                        <b>
                          {evaluation.test_start} — {evaluation.test_end}
                        </b>
                        <span className="banner-pill">Chronological split</span>
                      </div>
                      <div className="stats-grid">
                        <StatCard
                          icon={Activity}
                          label="MAE"
                          value={money(evaluation.mae)}
                          hint="Mean absolute error · lower is better"
                        />
                        <StatCard
                          icon={BarChart3}
                          label="RMSE"
                          value={money(evaluation.rmse)}
                          hint="Root mean squared error · lower is better"
                        />
                        <StatCard
                          icon={TrendingUp}
                          label="R² score"
                          value={Number(evaluation.r2).toFixed(4)}
                          hint="Explained variance score"
                        />
                        <StatCard
                          icon={ChartNoAxesCombined}
                          label="WAPE"
                          value={`${Number(evaluation.wape_percent).toFixed(2)}%`}
                          hint="Weighted absolute percentage error"
                        />
                      </div>
                      {evaluation.actual_predicted?.length > 0 && (
                        <Panel
                          title="Actual vs predicted demand"
                          subtitle="Daily total demand over the test period"
                        >
                          <div className="chart-wrap tall">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={evaluation.actual_predicted}>
                                <CartesianGrid
                                  strokeDasharray="3 3"
                                  vertical={false}
                                  stroke="#edf0f5"
                                />
                                <XAxis
                                  dataKey="date"
                                  tickFormatter={(v) => v.slice(5)}
                                  minTickGap={25}
                                />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Line
                                  type="monotone"
                                  dataKey="actual"
                                  name="Actual demand"
                                  stroke="#25a887"
                                  dot={false}
                                  strokeWidth={2}
                                />
                                <Line
                                  type="monotone"
                                  dataKey="predicted"
                                  name="Predicted demand"
                                  stroke="#7165f2"
                                  dot={false}
                                  strokeWidth={2}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </Panel>
                      )}
                      <div className="note-card">
                        <b>How to interpret</b>
                        <p>
                          R² is not the same as prediction accuracy. These
                          results come from a chronological holdout; actual
                          performance may differ in production. Forecasting
                          assumes non-demand inputs are known or held at recent
                          observed values.
                        </p>
                      </div>
                    </>
                  )}
                </>
              )}
            </>
          )}
          <footer className="footer">
            <span>DemandIQ · Retail Demand Forecasting</span>
            <span>React · FastAPI · scikit-learn</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
