"""Static dashboard contracts; Node executes the browser code with a minimal DOM."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_static_entrypoint():
    html = (ROOT / "docs/index.html").read_text()
    assert 'href="styles.css"' in html
    assert 'src="app.js"' in html
    assert 'role="status"' in html
    assert '<noscript>' in html
    assert 'for="start"' in html and 'for="size"' in html
    assert '<caption>' in html and 'scope="col"' in html
    assert 'https://' not in html and 'http://' not in html
    assert (ROOT / "docs/.nojekyll").exists()
    css = (ROOT / "docs/styles.css").read_text()
    assert ':focus-visible' in css and '@media' in css


HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Element {
  constructor() { this.children = []; this.value = ''; this.textContent = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this[key] = value; }
  addEventListener(event, callback) { this[event] = callback; }
}
const payload = JSON.parse(fs.readFileSync('docs/data/forecast.json'));
async function run(data, failure = false) {
  const elements = {};
  const document = {
    getElementById: id => elements[id] ||= new Element(),
    createElement: () => new Element(), createElementNS: () => new Element()
  };
  let requested;
  const context = vm.createContext({ document, console, URL,
    fetch: async url => { requested = url; return {ok: !failure, status: 404, json: async () => data}; }
  });
  vm.runInContext(fs.readFileSync('docs/app.js', 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requested, 'data/forecast.json');
  return elements;
}
(async () => {
  let ui = await run(payload);
  assert.match(ui.status.textContent, /Loaded/);
  assert.equal(ui.rows.children.length, 48);
  assert.equal(ui.chart.role, 'img');
  assert.equal(ui.chart['aria-labelledby'], 'chart-title chart-description');
  assert.equal(ui.chart.children.filter(c => c.role === 'presentation').length, 3);
  assert.match(ui.metrics.textContent, /164.87/);
  assert.match(ui.metrics.textContent, /826.63/);
  assert.match(ui.methodology.textContent, /observation-row shifts/);
  assert.match(ui.methodology.textContent, /prior observed targets/i);
  for (const key of ['dataset', 'model', 'split', 'features', 'dependencies', 'evaluation']) {
    assert.ok(ui.metadata.textContent.includes(JSON.stringify(payload[key], null, 2)));
  }
  ui.start.value = String(payload.records.length - 1);
  ui.start.change();
  assert.equal(ui.rows.children.length, 1);
  assert.ok(ui.rows.children[0].children[0].textContent.includes(payload.records.at(-1).timestamp));
  ui.start.value = '-200'; ui.size.value = '999999'; ui.size.change();
  assert.equal(ui.rows.children.length, 168);
  assert.equal(ui.start.value, '0');
  ui = await run({...payload, records: []});
  assert.match(ui.status.textContent, /No historical records/);
  assert.equal(ui.controls.disabled, true);
  ui = await run(payload, true);
  assert.match(ui.status.textContent, /Unable to load/);
  assert.equal(ui.controls.disabled, true);
  for (const bad of [{}, {...payload, schema_version: 2},
    {...payload, records: [{timestamp: 'bad', actual: null}]},
    {...payload, metrics: {}}]) {
    ui = await run(bad);
    assert.match(ui.status.textContent, /Unable to load/);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def test_dashboard_runtime_contract():
    result = subprocess.run(["node", "-e", HARNESS], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
