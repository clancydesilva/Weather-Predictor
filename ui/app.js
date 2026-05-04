/**
 * ui/app.js — Cork City Weather UI
 *
 * Fetches live forecast from the local FastAPI backend and renders:
 *   - Dynamic weather theme (5 states based on rain probability)
 *   - Animated rain effect for rainy conditions
 *   - Hero card with current conditions
 *   - Clothing recommendations
 *   - Stats grid (rain %, comfort, wind, humidity)
 *   - Rain events timeline
 *   - 24h hourly strip
 *   - Browser notifications (morning briefing + on reload)
 *
 * Auto-refreshes every 5 minutes.
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────────

const API_BASE        = '';           // same origin — served by FastAPI
const REFRESH_MS      = 5 * 60_000;  // 5 minutes
const NOTIF_KEY       = 'corkweather_notif_sent';
const NOTIF_ENABLED   = 'corkweather_notif_enabled';

// ── State ─────────────────────────────────────────────────────────────────────

let raindrops       = [];
let refreshTimer    = null;
let currentTheme    = 'mild';

// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  startClock();
  init();
  // Restore notification button state
  if (localStorage.getItem(NOTIF_ENABLED) === 'true') {
    document.getElementById('notifyBtn').textContent = 'Enabled ✓';
    document.getElementById('notifyBtn').classList.add('enabled');
  }
});

async function init() {
  showLoading(true);
  hideError();

  try {
    const [today, outfit, now] = await Promise.all([
      fetchJSON('/forecast/today'),
      fetchJSON('/forecast/outfit'),
      fetchJSON('/forecast/now'),
    ]);

    render(today, outfit, now);
    showLoading(false);

    // Schedule morning notification if enabled
    maybeNotifyMorning(today, outfit);

    // Schedule next refresh
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(init, REFRESH_MS);

  } catch (err) {
    console.error('Forecast fetch failed:', err);
    showLoading(false);
    showError();
  }
}

// ── Data fetching ─────────────────────────────────────────────────────────────

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${path}`);
  return res.json();
}

// ── Master render ─────────────────────────────────────────────────────────────

function render(today, outfit, nowHours) {
  const ds  = today.daily_summary;
  const now = nowHours[0] || today.hours[0];

  // 1. Determine theme from forecast data
  const theme = getTheme(ds.peak_rain_probability, ds.avg_comfort_score);
  applyTheme(theme);

  // 2. Sections
  renderHero(now, ds, today.station);
  renderOutfit(outfit);
  renderStats(ds, now);
  renderEvents([...today.onset_events, ...today.offset_events]);
  renderHourly(today.hours);
  updateLastUpdated(today.generated_at);
}

// ── Theme ─────────────────────────────────────────────────────────────────────

function getTheme(peakRain, avgComfort) {
  if (peakRain > 0.65) return 'heavy-rain';
  if (peakRain > 0.35) return 'light-rain';
  if (avgComfort < 5)  return 'overcast';
  if (avgComfort > 7.5) return 'pleasant';
  return 'mild';
}

function applyTheme(theme) {
  if (theme === currentTheme) return;
  currentTheme = theme;

  const body = document.body;
  body.className = `theme-${theme}`;

  // Rain drops
  const dropCount = {
    'heavy-rain': 90,
    'light-rain': 40,
    'overcast':    0,
    'mild':        0,
    'pleasant':    0,
  }[theme] || 0;

  createRaindrops(dropCount);
}

function createRaindrops(count) {
  const container = document.getElementById('rainContainer');
  container.innerHTML = '';
  raindrops = [];

  for (let i = 0; i < count; i++) {
    const drop = document.createElement('div');
    drop.className = 'raindrop';
    drop.style.left            = `${Math.random() * 110 - 5}vw`;
    drop.style.animationDuration  = `${0.4 + Math.random() * 0.9}s`;
    drop.style.animationDelay    = `${Math.random() * 2.5}s`;
    drop.style.height            = `${14 + Math.random() * 16}px`;
    drop.style.opacity           = 0.25 + Math.random() * 0.55;
    container.appendChild(drop);
    raindrops.push(drop);
  }
}

// ── Hero card ─────────────────────────────────────────────────────────────────

const CONDITION_META = {
  'heavy-rain':  { icon: '🌧️', label: 'Heavy Rain Expected' },
  'light-rain':  { icon: '🌦️', label: 'Light Rain Likely'   },
  'overcast':    { icon: '☁️',  label: 'Cloudy & Overcast'   },
  'mild':        { icon: '🌤️',  label: 'Mild Conditions'     },
  'pleasant':    { icon: '☀️',  label: 'Clear & Comfortable' },
};

function renderHero(now, ds, station) {
  const meta = CONDITION_META[currentTheme] || CONDITION_META['mild'];

  el('conditionIcon').textContent  = meta.icon;
  el('conditionBadge').textContent = meta.label;
  el('stationName').textContent    = station || 'Cork Airport';

  el('tempNow').textContent   = fmt1(now.temp_c);
  el('feelsLike').textContent = fmt1(now.feels_like_c);
  el('tempMax').textContent   = fmt1(ds.max_temp_c);
  el('tempMin').textContent   = fmt1(ds.min_temp_c);
  el('totalRain').textContent = ds.total_rainfall_mm.toFixed(1);
  el('rainHours').textContent = ds.rain_hours;
}

// ── Outfit card ───────────────────────────────────────────────────────────────

function renderOutfit(outfit) {
  el('outfitConfidence').textContent = outfit.confidence || '';

  // Clothing chips
  const container = el('outfitItems');
  container.innerHTML = '';
  (outfit.items || []).forEach((item, i) => {
    const chip = document.createElement('span');
    chip.className = 'outfit-chip';
    chip.style.animationDelay = `${i * 0.06}s`;
    chip.textContent = chipIcon(item) + ' ' + item;
    container.appendChild(chip);
  });

  // Alerts (umbrella inversion / waterproof)
  const alerts = el('outfitAlerts');
  alerts.innerHTML = '';
  if (outfit.waterproof) {
    alerts.appendChild(makeAlert('🌬️', 'Strong winds — umbrella may invert. Go waterproof.'));
  }
  if (outfit.umbrella_risk && !outfit.waterproof) {
    alerts.appendChild(makeAlert('☂️', 'Bring an umbrella — rain expected.'));
  }
}

function chipIcon(item) {
  const icons = {
    'Heavy coat': '🧥', 'Jacket': '🧥', 'Light jacket': '🧥',
    'Jumper': '🧣', 'Gloves': '🧤', 'Scarf': '🧣',
    'T-shirt': '👕', 'Umbrella': '☂️', 'Waterproof': '🌧️',
    'Wind-resistant layer': '🌬️',
  };
  for (const [key, icon] of Object.entries(icons)) {
    if (item.includes(key)) return icon;
  }
  return '👕';
}

function makeAlert(icon, text) {
  const div  = document.createElement('div');
  div.className = 'outfit-alert';
  const ico  = document.createElement('span');
  ico.className = 'outfit-alert-icon';
  ico.textContent = icon;
  const msg  = document.createElement('span');
  msg.textContent = text;
  div.appendChild(ico);
  div.appendChild(msg);
  return div;
}

// ── Stats grid ────────────────────────────────────────────────────────────────

function renderStats(ds, now) {
  // Rain probability
  const rainPct = Math.round(ds.peak_rain_probability * 100);
  el('statRain').textContent = `${rainPct}%`;
  el('rainBar').style.width  = `${rainPct}%`;

  // Comfort score with dots
  const comfort = parseFloat(ds.avg_comfort_score);
  el('statComfort').innerHTML = `${comfort.toFixed(1)}<span class="stat-unit">/10</span>`;
  renderComfortDots(comfort);

  // Wind
  el('statWind').textContent    = `${Math.round(now.wind_speed_kmh)} <span style="font-size:0.8rem;font-weight:400">km/h</span>`;
  el('statWind').innerHTML      = `${Math.round(now.wind_speed_kmh)}<span class="stat-unit"> km/h</span>`;
  el('statWindDir').textContent = windDir(now.wind_dir_deg || 0);

  // Humidity
  const hum = Math.round(now.humidity_pct || 0);
  el('statHumidity').textContent = `${hum}%`;
  el('humidBar').style.width     = `${hum}%`;
}

function renderComfortDots(score) {
  const container = el('comfortDots');
  container.innerHTML = '';
  const total = 10;
  const filled = Math.round(score);
  for (let i = 0; i < total; i++) {
    const dot = document.createElement('div');
    dot.className = 'comfort-dot' + (i < filled ? ' filled' : '');
    container.appendChild(dot);
  }
}

function windDir(deg) {
  const dirs = ['N','NE','E','SE','S','SW','W','NW'];
  return dirs[Math.round((deg % 360) / 45) % 8];
}

// ── Events timeline ───────────────────────────────────────────────────────────

function renderEvents(events) {
  const list = el('eventsList');
  list.innerHTML = '';

  if (!events || events.length === 0) {
    el('eventsCount').textContent = '';
    list.innerHTML = '<p class="no-events">No significant events predicted.</p>';
    return;
  }

  // Sort by datetime string
  const sorted = [...events].sort((a, b) => a.datetime.localeCompare(b.datetime));
  el('eventsCount').textContent = `${sorted.length} event${sorted.length !== 1 ? 's' : ''}`;

  sorted.forEach((e, i) => {
    const time = e.datetime.substring(11, 16); // "HH:MM"
    const conf = Math.round(e.confidence * 100);
    const type = e.event === 'onset' ? 'onset' : 'offset';
    const verb = type === 'onset' ? 'Rain starts' : 'Rain stops';

    const item = document.createElement('div');
    item.className = `event-item ${type}`;
    item.style.animationDelay = `${i * 0.06}s`;
    item.innerHTML = `
      <div class="event-time">${time}</div>
      <div class="event-body">
        <div class="event-type">${verb}</div>
        <div class="event-message">${e.message || `${verb} around ${time}`}</div>
      </div>
      <div class="event-confidence">${conf}%</div>
    `;
    list.appendChild(item);
  });
}

// ── Hourly strip ──────────────────────────────────────────────────────────────

function renderHourly(hours) {
  const strip = el('hourlyStrip');
  strip.innerHTML = '';

  (hours || []).forEach((h, i) => {
    const time = h.datetime.substring(11, 16);
    const prob = Math.round(h.rain_probability * 100);
    const isRain = h.rain_flag === 1;
    const icon = isRain ? '🌧️' : (prob > 30 ? '🌦️' : '⛅');

    const card = document.createElement('div');
    card.className = `hourly-card${isRain ? ' rain-hour' : ''}`;
    card.style.animationDelay = `${i * 0.02}s`;
    card.innerHTML = `
      <div class="hourly-time">${time}</div>
      <div class="hourly-icon">${icon}</div>
      <div class="hourly-temp">${fmt1(h.temp_c)}°</div>
      <div class="hourly-prob">${prob}%</div>
      <div class="hourly-bar-track">
        <div class="hourly-bar" style="width:${prob}%"></div>
      </div>
    `;
    strip.appendChild(card);
  });
}

// ── Notifications ─────────────────────────────────────────────────────────────

function toggleNotifications() {
  const btn = el('notifyBtn');
  if (Notification.permission === 'granted' &&
      localStorage.getItem(NOTIF_ENABLED) === 'true') {
    // Disable
    localStorage.removeItem(NOTIF_ENABLED);
    btn.textContent = 'Enable';
    btn.classList.remove('enabled');
  } else {
    requestNotification();
  }
}

async function requestNotification() {
  if (!('Notification' in window)) {
    alert('Your browser does not support notifications.');
    return;
  }
  const perm = await Notification.requestPermission();
  const btn  = el('notifyBtn');
  if (perm === 'granted') {
    localStorage.setItem(NOTIF_ENABLED, 'true');
    btn.textContent = 'Enabled ✓';
    btn.classList.add('enabled');
    new Notification('Cork Weather 🌤️', {
      body: 'Morning briefings are enabled. You\'ll get a daily forecast at 7am.',
      icon: '/ui/icon.png',
    });
  } else {
    btn.textContent = 'Blocked';
  }
}

function maybeNotifyMorning(today, outfit) {
  if (Notification.permission !== 'granted') return;
  if (localStorage.getItem(NOTIF_ENABLED) !== 'true') return;

  const now    = new Date();
  const hour   = now.getHours();
  const today_ = now.toDateString();
  const key    = `${NOTIF_KEY}_${today_}`;

  // Send once per day between 7am and 9am
  if (hour >= 7 && hour < 9 && !localStorage.getItem(key)) {
    localStorage.setItem(key, '1');
    const ds    = today.daily_summary;
    const items = (outfit.items || []).join(', ');
    const rain  = Math.round(ds.peak_rain_probability * 100);
    new Notification(`Cork Weather — ${now.toLocaleDateString('en-IE', { weekday: 'long' })}`, {
      body: `${Math.round(ds.min_temp_c)}–${Math.round(ds.max_temp_c)}°C · ${rain}% rain · Wear: ${items}`,
      icon: '/ui/icon.png',
    });
  }
}

// ── Clock ─────────────────────────────────────────────────────────────────────

function startClock() {
  const tick = () => {
    const now = new Date();
    el('liveClock').textContent =
      now.toLocaleTimeString('en-IE', { hour: '2-digit', minute: '2-digit', hour12: false });
  };
  tick();
  setInterval(tick, 1000);
}

function updateLastUpdated(generatedAt) {
  if (!generatedAt) return;
  const dt = new Date(generatedAt);
  const timeStr = dt.toLocaleTimeString('en-IE', { hour: '2-digit', minute: '2-digit', hour12: false });
  el('lastUpdated').innerHTML = `<span class="live-dot"></span>Updated ${timeStr}`;
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function showLoading(show) {
  const overlay = el('loadingOverlay');
  if (show) {
    overlay.classList.remove('hidden');
  } else {
    overlay.classList.add('hidden');
  }
}

function showError()  { el('errorOverlay').classList.remove('hidden'); }
function hideError()  { el('errorOverlay').classList.add('hidden');    }

function el(id) { return document.getElementById(id); }

function fmt1(val) {
  if (val == null) return '--';
  return parseFloat(val).toFixed(1);
}
