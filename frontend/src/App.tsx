import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'

import {
  createInstrument,
  getInstruments,
  getMarketData,
  getMarketDataSummary,
  updateInstrument,
} from './api'

import type {
  Instrument,
  InstrumentCreate,
  InstrumentUpdate,
  MarketData,
  MarketDataSummary,
} from './api'

type View = 'instruments' | 'market-data'

function App() {
  const [view, setView] = useState<View>('instruments')

  const [instruments, setInstruments] = useState<Instrument[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [showForm, setShowForm] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [formError, setFormError] = useState('')

  const [editingInstrument, setEditingInstrument] =
    useState<Instrument | null>(null)

  const [form, setForm] = useState<InstrumentCreate>({
    symbol: '',
    name: '',
    asset_type: 'stock',
    exchange: '',
    currency: '',
  })

  const [editForm, setEditForm] = useState<InstrumentUpdate>({
    name: '',
    asset_type: 'stock',
    exchange: '',
    currency: '',
    is_active: true,
  })

  const [selectedSymbol, setSelectedSymbol] = useState('')

  const [marketData, setMarketData] = useState<MarketData[]>([])
  const [marketSummary, setMarketSummary] =
    useState<MarketDataSummary | null>(null)

  const [marketLoading, setMarketLoading] = useState(false)
  const [marketError, setMarketError] = useState('')

  async function loadInstruments() {
    try {
      setLoading(true)
      setError('')

      const data = await getInstruments()
      setInstruments(data)

      if (!selectedSymbol && data.length > 0) {
        const activeInstrument =
          data.find((instrument) => instrument.is_active) ?? data[0]

        setSelectedSymbol(activeInstrument.symbol)
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to load instruments',
      )
    } finally {
      setLoading(false)
    }
  }

  async function loadMarketData(symbol: string) {
    if (!symbol) {
      setMarketData([])
      setMarketSummary(null)
      return
    }

    try {
      setMarketLoading(true)
      setMarketError('')

      const [summary, history] = await Promise.all([
        getMarketDataSummary(symbol),
        getMarketData(symbol),
      ])

      setMarketSummary(summary)
      setMarketData(history)
    } catch (err) {
      setMarketData([])
      setMarketSummary(null)

      setMarketError(
        err instanceof Error
          ? err.message
          : 'Unable to load market data',
      )
    } finally {
      setMarketLoading(false)
    }
  }

  useEffect(() => {
    loadInstruments()
  }, [])

  useEffect(() => {
    if (view === 'market-data' && selectedSymbol) {
      loadMarketData(selectedSymbol)
    }
  }, [view, selectedSymbol])

  function updateField(
    field: keyof InstrumentCreate,
    value: string,
  ) {
    setForm((current) => ({
      ...current,
      [field]: value,
    }))
  }

  function openEditForm(instrument: Instrument) {
    setEditingInstrument(instrument)
    setFormError('')

    setEditForm({
      name: instrument.name,
      asset_type: instrument.asset_type,
      exchange: instrument.exchange || '',
      currency: instrument.currency || '',
      is_active: instrument.is_active,
    })
  }

  function closeEditForm() {
    setEditingInstrument(null)
    setFormError('')
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    try {
      setSubmitting(true)
      setFormError('')

      await createInstrument({
        symbol: form.symbol.trim().toUpperCase(),
        name: form.name.trim(),
        asset_type: form.asset_type.trim().toLowerCase(),
        exchange: form.exchange?.trim() || undefined,
        currency: form.currency?.trim().toUpperCase() || undefined,
      })

      setForm({
        symbol: '',
        name: '',
        asset_type: 'stock',
        exchange: '',
        currency: '',
      })

      setShowForm(false)
      await loadInstruments()
    } catch (err) {
      setFormError(
        err instanceof Error
          ? err.message
          : 'Unable to create instrument',
      )
    } finally {
      setSubmitting(false)
    }
  }

  async function handleEditSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault()

    if (!editingInstrument) {
      return
    }

    try {
      setSubmitting(true)
      setFormError('')

      await updateInstrument(editingInstrument.id, {
        name: editForm.name?.trim(),
        asset_type: editForm.asset_type?.trim().toLowerCase(),
        exchange: editForm.exchange?.trim() || undefined,
        currency: editForm.currency?.trim().toUpperCase() || undefined,
        is_active: editForm.is_active,
      })

      closeEditForm()
      await loadInstruments()
    } catch (err) {
      setFormError(
        err instanceof Error
          ? err.message
          : 'Unable to update instrument',
      )
    } finally {
      setSubmitting(false)
    }
  }

  function formatPrice(value: number) {
    return value.toFixed(2)
  }

  function formatPercent(value: number) {
    const sign = value > 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}%`
  }

  function formatNumber(value: number) {
    return new Intl.NumberFormat('en-US').format(value)
  }

  function formatDate(value: string) {
    return new Date(value).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  const selectedInstrument = instruments.find(
    (instrument) => instrument.symbol === selectedSymbol,
  )

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">AQ</div>

          <div>
            <div className="brand-name">AegisQuant</div>
            <div className="brand-subtitle">
              AI Trading Platform
            </div>
          </div>
        </div>

        <div className="backend-status">
          <span className="status-dot" />
          Backend ready
        </div>
      </header>

      <nav className="main-nav">
        <button
          type="button"
          className={view === 'instruments' ? 'nav-button active' : 'nav-button'}
          onClick={() => setView('instruments')}
        >
          Instruments
        </button>

        <button
          type="button"
          className={view === 'market-data' ? 'nav-button active' : 'nav-button'}
          onClick={() => setView('market-data')}
        >
          Market Data
        </button>
      </nav>

      <main className="content">
        {view === 'instruments' && (
          <>
            <section className="page-heading">
              <div>
                <p className="eyebrow">INSTRUMENT MANAGEMENT</p>
                <h1>Instruments</h1>
                <p className="page-description">
                  Manage the financial instruments available to AegisQuant.
                </p>
              </div>

              <button
                className="primary-button"
                type="button"
                onClick={() => {
                  setShowForm((current) => !current)
                  setFormError('')
                }}
              >
                {showForm ? 'Cancel' : '+ Add Instrument'}
              </button>
            </section>

            {showForm && (
              <section className="form-card">
                <div className="form-header">
                  <div>
                    <p className="eyebrow">NEW INSTRUMENT</p>
                    <h2>Add Instrument</h2>
                  </div>
                </div>

                <form onSubmit={handleSubmit}>
                  <div className="form-grid">
                    <label>
                      <span>Symbol</span>
                      <input
                        required
                        value={form.symbol}
                        onChange={(event) =>
                          updateField('symbol', event.target.value)
                        }
                        placeholder="AAPL"
                      />
                    </label>

                    <label>
                      <span>Name</span>
                      <input
                        required
                        value={form.name}
                        onChange={(event) =>
                          updateField('name', event.target.value)
                        }
                        placeholder="Apple Inc."
                      />
                    </label>

                    <label>
                      <span>Asset Type</span>
                      <select
                        value={form.asset_type}
                        onChange={(event) =>
                          updateField('asset_type', event.target.value)
                        }
                      >
                        <option value="stock">Stock</option>
                        <option value="etf">ETF</option>
                        <option value="forex">Forex</option>
                        <option value="crypto">Crypto</option>
                        <option value="index">Index</option>
                      </select>
                    </label>

                    <label>
                      <span>Exchange</span>
                      <input
                        value={form.exchange}
                        onChange={(event) =>
                          updateField('exchange', event.target.value)
                        }
                        placeholder="NASDAQ"
                      />
                    </label>

                    <label>
                      <span>Currency</span>
                      <input
                        value={form.currency}
                        onChange={(event) =>
                          updateField('currency', event.target.value)
                        }
                        placeholder="USD"
                      />
                    </label>
                  </div>

                  {formError && (
                    <div className="form-error">{formError}</div>
                  )}

                  <div className="form-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => {
                        setShowForm(false)
                        setFormError('')
                      }}
                    >
                      Cancel
                    </button>

                    <button
                      type="submit"
                      className="primary-button"
                      disabled={submitting}
                    >
                      {submitting ? 'Adding...' : 'Add Instrument'}
                    </button>
                  </div>
                </form>
              </section>
            )}

            {editingInstrument && (
              <section className="form-card">
                <div className="form-header">
                  <div>
                    <p className="eyebrow">EDIT INSTRUMENT</p>
                    <h2>{editingInstrument.symbol}</h2>
                  </div>
                </div>

                <form onSubmit={handleEditSubmit}>
                  <div className="form-grid">
                    <label>
                      <span>Symbol</span>
                      <input
                        value={editingInstrument.symbol}
                        disabled
                      />
                    </label>

                    <label>
                      <span>Name</span>
                      <input
                        required
                        value={editForm.name || ''}
                        onChange={(event) =>
                          setEditForm((current) => ({
                            ...current,
                            name: event.target.value,
                          }))
                        }
                      />
                    </label>

                    <label>
                      <span>Asset Type</span>
                      <select
                        value={editForm.asset_type || 'stock'}
                        onChange={(event) =>
                          setEditForm((current) => ({
                            ...current,
                            asset_type: event.target.value,
                          }))
                        }
                      >
                        <option value="stock">Stock</option>
                        <option value="etf">ETF</option>
                        <option value="forex">Forex</option>
                        <option value="crypto">Crypto</option>
                        <option value="index">Index</option>
                      </select>
                    </label>

                    <label>
                      <span>Exchange</span>
                      <input
                        value={editForm.exchange || ''}
                        onChange={(event) =>
                          setEditForm((current) => ({
                            ...current,
                            exchange: event.target.value,
                          }))
                        }
                      />
                    </label>

                    <label>
                      <span>Currency</span>
                      <input
                        value={editForm.currency || ''}
                        onChange={(event) =>
                          setEditForm((current) => ({
                            ...current,
                            currency: event.target.value,
                          }))
                        }
                      />
                    </label>

                    <label className="checkbox-field">
                      <span>Status</span>

                      <div className="checkbox-row">
                        <input
                          type="checkbox"
                          checked={editForm.is_active ?? true}
                          onChange={(event) =>
                            setEditForm((current) => ({
                              ...current,
                              is_active: event.target.checked,
                            }))
                          }
                        />

                        <span>Instrument is active</span>
                      </div>
                    </label>
                  </div>

                  {formError && (
                    <div className="form-error">{formError}</div>
                  )}

                  <div className="form-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={closeEditForm}
                    >
                      Cancel
                    </button>

                    <button
                      type="submit"
                      className="primary-button"
                      disabled={submitting}
                    >
                      {submitting ? 'Saving...' : 'Save Changes'}
                    </button>
                  </div>
                </form>
              </section>
            )}

            <section className="table-card">
              <div className="table-header">
                <div>
                  <span className="instrument-count">
                    {instruments.length}
                  </span>
                  <span className="instrument-count-label">
                    TOTAL
                  </span>
                </div>

                <button
                  type="button"
                  className="refresh-button"
                  onClick={loadInstruments}
                  disabled={loading}
                >
                  {loading ? 'Loading...' : 'Refresh'}
                </button>
              </div>

              {loading && (
                <div className="table-message">
                  Loading instruments...
                </div>
              )}

              {!loading && error && (
                <div className="table-message error-message">
                  <strong>Unable to load instruments</strong>
                  <span>{error}</span>
                  <span>
                    Make sure the FastAPI backend is running on port 8000.
                  </span>
                </div>
              )}

              {!loading && !error && instruments.length === 0 && (
                <div className="table-message">
                  No instruments have been added yet.
                </div>
              )}

              {!loading && !error && instruments.length > 0 && (
                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Name</th>
                        <th>Asset Type</th>
                        <th>Exchange</th>
                        <th>Currency</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>

                    <tbody>
                      {instruments.map((instrument) => (
                        <tr key={instrument.id}>
                          <td className="symbol-cell">
                            {instrument.symbol}
                          </td>

                          <td>{instrument.name}</td>
                          <td>{instrument.asset_type}</td>
                          <td>{instrument.exchange || '—'}</td>
                          <td>{instrument.currency || '—'}</td>

                          <td>
                            <span
                              className={
                                instrument.is_active
                                  ? 'status-badge active'
                                  : 'status-badge inactive'
                              }
                            >
                              {instrument.is_active
                                ? 'Active'
                                : 'Inactive'}
                            </span>
                          </td>

                          <td>
                            <button
                              type="button"
                              className="edit-button"
                              onClick={() =>
                                openEditForm(instrument)
                              }
                            >
                              Edit
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </>
        )}

        {view === 'market-data' && (
          <>
            <section className="page-heading">
              <div>
                <p className="eyebrow">MARKET DATA</p>
                <h1>Market Data</h1>
                <p className="page-description">
                  Monitor historical market data available to AegisQuant.
                </p>
              </div>

              <div className="market-controls">
                <select
                  className="instrument-select"
                  value={selectedSymbol}
                  onChange={(event) =>
                    setSelectedSymbol(event.target.value)
                  }
                >
                  {instruments
                    .filter((instrument) => instrument.is_active)
                    .map((instrument) => (
                      <option
                        key={instrument.id}
                        value={instrument.symbol}
                      >
                        {instrument.symbol} — {instrument.name}
                      </option>
                    ))}
                </select>

                <button
                  type="button"
                  className="refresh-button"
                  onClick={() =>
                    selectedSymbol &&
                    loadMarketData(selectedSymbol)
                  }
                  disabled={marketLoading || !selectedSymbol}
                >
                  {marketLoading ? 'Loading...' : 'Refresh'}
                </button>
              </div>
            </section>

            {marketLoading && (
              <div className="table-card">
                <div className="table-message">
                  Loading market data...
                </div>
              </div>
            )}

            {!marketLoading && marketError && (
              <div className="table-card">
                <div className="table-message error-message">
                  <strong>Unable to load market data</strong>
                  <span>{marketError}</span>
                </div>
              </div>
            )}

            {!marketLoading && !marketError && marketSummary && (
              <>
                <section className="market-overview">
                  <div className="metric-card">
                    <span>Latest Close</span>
                    <strong>
                      {formatPrice(marketSummary.latest_close)}
                    </strong>
                    <small>
                      {selectedInstrument?.currency || 'USD'}
                    </small>
                  </div>

                  <div className="metric-card">
                    <span>Change</span>
                    <strong
                      className={
                        marketSummary.change >= 0
                          ? 'positive-value'
                          : 'negative-value'
                      }
                    >
                      {marketSummary.change >= 0 ? '+' : ''}
                      {formatPrice(marketSummary.change)}
                    </strong>
                    <small
                      className={
                        marketSummary.change_percent >= 0
                          ? 'positive-value'
                          : 'negative-value'
                      }
                    >
                      {formatPercent(
                        marketSummary.change_percent,
                      )}
                    </small>
                  </div>

                  <div className="metric-card">
                    <span>Period High</span>
                    <strong>
                      {formatPrice(marketSummary.high)}
                    </strong>
                    <small>
                      {formatDate(marketSummary.last_timestamp)}
                    </small>
                  </div>

                  <div className="metric-card">
                    <span>Period Low</span>
                    <strong>
                      {formatPrice(marketSummary.low)}
                    </strong>
                    <small>
                      {marketSummary.data_points} data points
                    </small>
                  </div>
                </section>

                <section className="table-card">
                  <div className="table-header">
                    <div>
                      <span className="section-title">
                        {marketSummary.symbol}
                      </span>

                      <span className="instrument-count-label">
                        HISTORICAL DATA
                      </span>
                    </div>

                    <span className="data-range">
                      {formatDate(marketSummary.first_timestamp)}
                      {' → '}
                      {formatDate(marketSummary.last_timestamp)}
                    </span>
                  </div>

                  {marketData.length === 0 ? (
                    <div className="table-message">
                      No market data available.
                    </div>
                  ) : (
                    <div className="table-wrapper">
                      <table>
                        <thead>
                          <tr>
                            <th>Date</th>
                            <th>Open</th>
                            <th>High</th>
                            <th>Low</th>
                            <th>Close</th>
                            <th>Volume</th>
                          </tr>
                        </thead>

                        <tbody>
                          {marketData.map((row) => (
                            <tr key={row.id}>
                              <td>{formatDate(row.timestamp)}</td>
                              <td>{formatPrice(row.open)}</td>
                              <td>{formatPrice(row.high)}</td>
                              <td>{formatPrice(row.low)}</td>
                              <td className="symbol-cell">
                                {formatPrice(row.close)}
                              </td>
                              <td>
                                {row.volume === null
                                  ? '—'
                                  : formatNumber(row.volume)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>
              </>
            )}
          </>
        )}
      </main>

      <footer>© 2026 AegisQuant</footer>
    </div>
  )
}

export default App
