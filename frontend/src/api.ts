const API_BASE_URL = 'http://127.0.0.1:8000'

export type Instrument = {
  id: number
  symbol: string
  name: string
  asset_type: string
  exchange: string | null
  currency: string | null
  is_active: boolean
  created_at: string
}

export type InstrumentCreate = {
  symbol: string
  name: string
  asset_type: string
  exchange?: string
  currency?: string
}

export type InstrumentUpdate = {
  name?: string
  asset_type?: string
  exchange?: string
  currency?: string
  is_active?: boolean
}

export type MarketData = {
  id: number
  instrument_id: number
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number | null
}

export type MarketDataSummary = {
  symbol: string
  data_points: number
  first_timestamp: string
  last_timestamp: string
  first_open: number
  latest_close: number
  high: number
  low: number
  change: number
  change_percent: number
}

async function apiRequest<T>(
  url: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers || {}),
    },
    ...options,
  })

  if (!response.ok) {
    let message = `API request failed with status ${response.status}`

    try {
      const body = await response.json()

      if (typeof body?.detail === 'string') {
        message = body.detail
      } else if (Array.isArray(body?.detail)) {
        message = body.detail
          .map((item: { msg?: string }) => item.msg)
          .join(', ')
      }
    } catch {
      // Keep the default error message.
    }

    throw new Error(message)
  }

  return response.json()
}

export async function getInstruments(): Promise<Instrument[]> {
  return apiRequest<Instrument[]>(
    `${API_BASE_URL}/instruments/`,
  )
}

export async function createInstrument(
  instrument: InstrumentCreate,
): Promise<Instrument> {
  return apiRequest<Instrument>(
    `${API_BASE_URL}/instruments/`,
    {
      method: 'POST',
      body: JSON.stringify(instrument),
    },
  )
}

export async function updateInstrument(
  instrumentId: number,
  instrument: InstrumentUpdate,
): Promise<Instrument> {
  return apiRequest<Instrument>(
    `${API_BASE_URL}/instruments/${instrumentId}`,
    {
      method: 'PATCH',
      body: JSON.stringify(instrument),
    },
  )
}

export async function getMarketData(
  symbol: string,
): Promise<MarketData[]> {
  return apiRequest<MarketData[]>(
    `${API_BASE_URL}/market-data/symbol/${encodeURIComponent(symbol)}`,
  )
}

export async function getLatestMarketData(
  symbol: string,
  limit = 100,
): Promise<MarketData[]> {
  return apiRequest<MarketData[]>(
    `${API_BASE_URL}/market-data/latest/${encodeURIComponent(symbol)}?limit=${limit}`,
  )
}

export async function getMarketDataSummary(
  symbol: string,
): Promise<MarketDataSummary> {
  return apiRequest<MarketDataSummary>(
    `${API_BASE_URL}/market-data/summary/${encodeURIComponent(symbol)}`,
  )
}
