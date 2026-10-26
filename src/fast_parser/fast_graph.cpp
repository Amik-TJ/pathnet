#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using i64 = int64_t;
using std::string;
using std::string_view;
using std::vector;

enum { FP_OK = 0, FP_UNSUPPORTED = 1, FP_FALLBACK = 2 };

struct LibSpec {
  i64 n_cells;
  const char* cell_names;
  const int8_t* cell_flags;
  const i64* pin_ptr;
  const char* pin_names;
  const i64* pin_vocab;
  const int8_t* pin_dir;
  const int8_t* pin_role;
  const double* pin_cap_rise;
  const double* pin_cap_fall;
  const double* pin_max_tr;
  const double* pin_max_cap;
  const int8_t* pin_const_init;
  const i64* pin_func;
  const i64* arc_ptr;
  const i64* arc_from;
  const i64* arc_to;
  const i64* arc_type;
  const i64* arc_sense;
  const int8_t* arc_rf;
  const i64* arc_when_vocab;
  const i64* arc_key;
  const i64* arc_when_expr;
  const i64* arc_collapse_slot;
  const i64* chk_ptr;
  const i64* chk_clk;
  const i64* chk_d;
  const i64* chk_bank;
  i64 n_expr;
  const i64* expr_ptr;
  const i64* expr_code;
  const i64* expr_slot_ptr;
  const i64* expr_slot_pin;
};

struct Lib {
  i64 n_cells = 0;
  vector<string> cell_name;
  std::unordered_map<string, int> cell_id;
  vector<int8_t> cell_flags;
  vector<i64> pin_ptr;
  vector<string> pin_name;
  vector<std::unordered_map<string, int>> pin_index;
  vector<i64> pin_vocab;
  vector<int8_t> pin_dir, pin_role, pin_const_init;
  vector<double> pin_cap_rise, pin_cap_fall, pin_max_tr, pin_max_cap;
  vector<i64> pin_func;
  vector<i64> arc_ptr, arc_from, arc_to, arc_type, arc_sense, arc_when_vocab, arc_key, arc_when_expr, arc_collapse_slot;
  vector<int8_t> arc_rf;
  vector<i64> chk_ptr, chk_clk, chk_d, chk_bank;
  vector<i64> expr_ptr, expr_code, expr_slot_ptr, expr_slot_pin;
};

template <class T>
static vector<T> cp(const T* p, i64 n) { return vector<T>(p, p + n); }

static inline bool ws(unsigned char c) { return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31); }
static inline bool alpha_(unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; }
static inline bool digit(unsigned char c) { return c >= '0' && c <= '9'; }
static inline bool word(unsigned char c) { return alpha_(c) || digit(c); }

static string_view strip(string_view s) {
  size_t a = 0, b = s.size();
  while (a < b && ws(s[a])) a++;
  while (b > a && ws(s[b - 1])) b--;
  return s.substr(a, b - a);
}
static bool starts(string_view s, string_view p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }

static string canon(string_view name) {
  name = strip(name);
  if (!name.empty() && name[0] == '\\') {
    name = strip(name.substr(1));
    size_t i = 0;
    while (i < name.size() && !ws(name[i])) i++;
    if (i > 0 && i < name.size()) {
      size_t j = i;
      while (j < name.size() && ws(name[j])) j++;
      if (j < name.size() && name[j] == '[') {
        size_t k = j + 1;
        while (k < name.size() && digit(name[k])) k++;
        if (k > j + 1 && k + 1 == name.size() && name[k] == ']') {
          string out(name.substr(0, i));
          out += '[';
          out += name.substr(j + 1, k - j - 1);
          out += ']';
          return out;
        }
      }
    }
  }
  return string(name);
}

static void split_names(string_view t, vector<string_view>& out) {
  out.clear();
  size_t a = 0;
  while (true) {
    size_t b = t.find(',', a);
    string_view part = strip(t.substr(a, b == string_view::npos ? string_view::npos : b - a));
    if (!part.empty()) out.push_back(part);
    if (b == string_view::npos) break;
    a = b + 1;
  }
}

static bool parse_int(string_view s, size_t& i, i64& v) {
  size_t j = i;
  bool neg = false;
  if (j < s.size() && s[j] == '-') { neg = true; j++; }
  size_t d0 = j;
  while (j < s.size() && digit(s[j])) j++;
  if (j == d0 || j - d0 > 18) return false;
  i64 x = 0;
  for (size_t k = d0; k < j; k++) x = x * 10 + (s[k] - '0');
  v = neg ? -x : x;
  i = j;
  return true;
}

static bool parse_range(string_view s, size_t& i, i64& a, i64& b) {
  if (i >= s.size() || s[i] != '[') return false;
  size_t j = i + 1;
  while (j < s.size() && ws(s[j])) j++;
  if (!parse_int(s, j, a)) return false;
  while (j < s.size() && ws(s[j])) j++;
  if (j >= s.size() || s[j] != ':') return false;
  j++;
  while (j < s.size() && ws(s[j])) j++;
  if (!parse_int(s, j, b)) return false;
  while (j < s.size() && ws(s[j])) j++;
  if (j >= s.size() || s[j] != ']') return false;
  i = j + 1;
  return true;
}

static string strip_pairs(const string& t, const char* open, const char* close) {
  string out;
  out.reserve(t.size());
  size_t i = 0;
  while (true) {
    size_t p = t.find(open, i);
    if (p == string::npos) break;
    size_t q = t.find(close, p + 2);
    if (q == string::npos) break;
    out.append(t, i, p - i);
    out += ' ';
    i = q + 2;
  }
  out.append(t, i, string::npos);
  return out;
}

static string strip_line_comments(const string& t) {
  string out;
  out.reserve(t.size());
  size_t i = 0;
  while (true) {
    size_t p = t.find("//", i);
    if (p == string::npos) break;
    size_t q = t.find('\n', p + 2);
    if (q == string::npos) q = t.size();
    out.append(t, i, p - i);
    out += ' ';
    i = q;
  }
  out.append(t, i, string::npos);
  return out;
}

static bool read_file(const char* path, string& out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize(n > 0 ? n : 0);
  size_t got = n > 0 ? fread(&out[0], 1, n, f) : 0;
  fclose(f);
  return (long)got == n;
}

static bool read_gz(const char* path, string& out) {
  gzFile f = gzopen(path, "rb");
  if (!f) return false;
  gzbuffer(f, 1 << 20);
  out.clear();
  char buf[1 << 16];
  int n;
  while ((n = gzread(f, buf, sizeof buf)) > 0) out.append(buf, n);
  bool ok = n == 0;
  gzclose(f);
  return ok;
}

static bool plain_ascii(const string& s) {
  for (unsigned char c : s)
    if (c >= 0x80 || c == '\r' || c == 0) return false;
  return true;
}

static double pairwise(const double* a, i64 n) {
  if (n < 8) {
    double res = -0.0;
    for (i64 i = 0; i < n; i++) res += a[i];
    return res;
  } else if (n <= 128) {
    double r[8];
    for (int k = 0; k < 8; k++) r[k] = a[k];
    i64 i;
    for (i = 8; i < n - (n % 8); i += 8)
      for (int k = 0; k < 8; k++) r[k] += a[i + k];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  } else {
    i64 n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
  }
}
static double np_sum(const double* a, i64 n) { return 0.0 + pairwise(a, n); }

struct Result {
  int status = FP_FALLBACK;
  string message;
  i64 N = 0, n_ports = 0, n_inst = 0;
  string pin_blob, inst_blob, master_blob, top;
  vector<i64> x_cat;
  vector<float> x_num;
  vector<float> x_erc;
  vector<i64> pin_instance, level;
  vector<i64> net_src, net_dst;
  vector<i64> a_src, a_dst, a_type, a_sense, a_when, a_key, a_inst;
  vector<uint8_t> a_rf, a_when_present, a_enabled;
  vector<i64> c_src, c_dst, c_bank;
  vector<i64> partial;
  vector<i64> pm_master, pm_count;
  i64 nets = 0, unloaded = 0, port_to_port = 0, when_disabled = 0, default_disabled = 0, collapsed = 0;
  i64 escaped = 0, n_masters = 0, n_registers = 0, tie_cells = 0, n_const = 0;
  int has_labels = 0;
  vector<i64> y_path, y_path_rf, y_worst, y_ep_index, y_ep_path, y_ep_ptr, y_ep_rf;
  vector<float> y_path_arr, y_ep_slack, y_ep_arr, y_ep_req, y_ep_path_arr;
  double worst_slack = 0, first_slack = 0;
};

static Result* fallback(Result* r, const string& why) {
  r->status = FP_FALLBACK;
  r->message = why;
  return r;
}

static int eval_expr(const Lib& L, i64 e, const int8_t* slotval) {
  int8_t st[256];
  int sp = 0;
  for (i64 k = L.expr_ptr[e]; k < L.expr_ptr[e + 1]; k++) {
    i64 c = L.expr_code[k];
    if (c >= 0) {
      st[sp++] = slotval[c];
    } else if (c == -1) {
      int8_t v = st[sp - 1];
      st[sp - 1] = v == 2 ? 2 : 1 - v;
    } else {
      int8_t b = st[--sp], a = st[sp - 1], v;
      if (c == -2) v = (a == 0 || b == 0) ? 0 : ((a == 2 || b == 2) ? 2 : 1);
      else if (c == -3) v = (a == 1 || b == 1) ? 1 : ((a == 2 || b == 2) ? 2 : 0);
      else v = (a == 2 || b == 2) ? 2 : (a ^ b);
      st[sp - 1] = v;
    }
  }
  return st[0];
}

static void fill_slots(const Lib& L, i64 e, const int8_t* pinval, int8_t* slotval) {
  for (i64 s = L.expr_slot_ptr[e]; s < L.expr_slot_ptr[e + 1]; s++) {
    i64 p = L.expr_slot_pin[s];
    slotval[s - L.expr_slot_ptr[e]] = p >= 0 ? pinval[p] : 2;
  }
}

static bool depends_on(const Lib& L, i64 e, i64 var, const int8_t* base) {
  i64 ns = L.expr_slot_ptr[e + 1] - L.expr_slot_ptr[e];
  int8_t v[128];
  vector<int> freev;
  for (i64 s = 0; s < ns; s++) {
    v[s] = base[s];
    if (s != var && base[s] == 2) freev.push_back((int)s);
  }
  for (i64 bits = 0; bits < (i64(1) << freev.size()); bits++) {
    for (size_t i = 0; i < freev.size(); i++) v[freev[i]] = (bits >> i) & 1;
    v[var] = 0;
    int v0 = eval_expr(L, e, v);
    v[var] = 1;
    int v1 = eval_expr(L, e, v);
    if (v0 != v1) return true;
  }
  return false;
}

struct Port { string name; int dir; };
struct Conn { int pin; int net; };
struct Inst { string name; string master; int cell; bool escaped; i64 c0, c1; };

struct SdcSpec {
  const char* top;
  const char* clock_port;
  i64 n_in;
  const char* in_names;
  const double* in_vals;
  i64 n_out;
  const char* out_names;
  const double* out_vals;
};

struct Hash {
  size_t operator()(string_view s) const noexcept { return std::hash<string_view>()(s); }
};

struct Ticker {
  bool on = getenv("FP_TIMING") != nullptr;
  std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
  void operator()(const char* what) {
    if (!on) return;
    auto now = std::chrono::steady_clock::now();
    fprintf(stderr, "  [cpp] %-28s %8.2f ms\n", what, std::chrono::duration<double, std::milli>(now - t).count());
    t = now;
  }
};

static Result* build(const Lib& L, const char* netlist_path, const SdcSpec* sdc, const char* label_dir) {
  Result* R = new Result();
  Ticker tick;
  string raw;
  if (!read_file(netlist_path, raw)) return fallback(R, "cannot read netlist");
  if (!plain_ascii(raw)) return fallback(R, "netlist not plain ASCII");
  tick("read netlist");
  string text = strip_line_comments(strip_pairs(strip_pairs(raw, "(*", "*)"), "/*", "*/"));
  raw.clear();
  raw.shrink_to_fit();
  tick("strip comments/attributes");

  string top;
  bool have_top = false;
  vector<string> header;
  vector<Port> ports;
  std::unordered_map<string, std::pair<i64, i64>> bus_range;
  vector<Inst> insts;
  vector<Conn> conns;
  vector<string> conn_net_name;
  std::unordered_map<string, int> conn_net_id;
  std::unordered_set<string> inst_seen;
  std::unordered_set<string> declared_bases;
  vector<string_view> names, inst_pins;

  string_view T(text);
  size_t a = 0;
  while (a <= T.size()) {
    size_t b = T.find(';', a);
    if (b == string_view::npos) b = T.size();
    string_view s = strip(T.substr(a, b - a));
    a = b + 1;
    if (s.empty()) continue;
    if (starts(s, "module")) {
      size_t i = 6;
      if (i >= s.size() || !ws(s[i])) return fallback(R, "module header");
      while (i < s.size() && ws(s[i])) i++;
      size_t r0 = i, r1 = i;
      while (r1 < s.size() && !ws(s[r1])) r1++;
      size_t q = s.size() - 1;
      if (r1 == r0 || s[q] != ')') return fallback(R, "module header");
      bool ok = false;
      size_t k, open = 0;
      for (k = r1 - r0; k >= 1; k--) {
        size_t p = r0 + k;
        if (k == r1 - r0) while (p < s.size() && ws(s[p])) p++;
        if (p < q && s[p] == '(') { ok = true; open = p; break; }
      }
      if (!ok) return fallback(R, "module header");
      top = canon(s.substr(r0, k));
      have_top = true;
      header.clear();
      split_names(s.substr(open + 1, q - open - 1), names);
      for (auto& n : names) header.push_back(canon(n));
      continue;
    }
    if (starts(s, "endmodule")) continue;
    if (starts(s, "wire") || starts(s, "reg")) {
      size_t i = starts(s, "wire") ? 4 : 3;
      while (i < s.size() && ws(s[i])) i++;
      i64 lo, hi;
      if (!parse_range(s, i, hi, lo)) continue;
      while (i < s.size() && ws(s[i])) i++;
      if (i >= s.size()) continue;
      split_names(s.substr(i), names);
      for (auto& n : names) bus_range.emplace(canon(n), std::make_pair(std::min(hi, lo), std::max(hi, lo)));
      continue;
    }
    if (starts(s, "assign")) return fallback(R, "assign");
    if (starts(s, "input") || starts(s, "output") || starts(s, "inout")) {
      if (starts(s, "inout")) return fallback(R, "inout");
      int dir = starts(s, "input") ? 0 : 1;
      size_t i = dir == 0 ? 5 : 6;
      if (i >= s.size() || !ws(s[i])) return fallback(R, "declaration without whitespace");
      while (i < s.size() && ws(s[i])) i++;
      bool bus = false;
      i64 hi = 0, lo = 0;
      if (i < s.size() && s[i] == '[') {
        if (!parse_range(s, i, hi, lo)) return fallback(R, "declaration range");
        bus = true;
        while (i < s.size() && ws(s[i])) i++;
      }
      if (i >= s.size()) return fallback(R, "declaration without names");
      split_names(s.substr(i), names);
      for (auto& n : names) {
        string nm = canon(n);
        declared_bases.insert(nm);
        if (!bus) {
          ports.push_back({nm, dir});
        } else {
          i64 mn = std::min(hi, lo), mx = std::max(hi, lo);
          bus_range.emplace(nm, std::make_pair(mn, mx));
          for (i64 x = mn; x <= mx; x++) ports.push_back({nm + "[" + std::to_string(x) + "]", dir});
        }
      }
      continue;
    }
    size_t i = 0;
    if (!alpha_(s[0])) return fallback(R, "unrecognised statement");
    while (i < s.size() && (word(s[i]) || s[i] == '$')) i++;
    string_view master = s.substr(0, i);
    if (i >= s.size() || !ws(s[i])) return fallback(R, "instance master");
    while (i < s.size() && ws(s[i])) i++;
    size_t n0 = i;
    string_view name_raw;
    if (i < s.size() && s[i] == '\\') {
      size_t j = i + 1;
      while (j < s.size() && !ws(s[j])) j++;
      if (j == i + 1 || j >= s.size()) return fallback(R, "escaped instance name");
      name_raw = s.substr(n0, j + 1 - n0);
      i = j + 1;
    } else if (i < s.size() && alpha_(s[i])) {
      while (i < s.size() && (word(s[i]) || s[i] == '$' || s[i] == '.' || s[i] == '[' || s[i] == ']')) i++;
      name_raw = s.substr(n0, i - n0);
    } else {
      return fallback(R, "instance name");
    }
    while (i < s.size() && ws(s[i])) i++;
    if (i >= s.size() || s[i] != '(' || s.back() != ')' || i + 1 > s.size() - 1) return fallback(R, "instance body");
    string_view body = s.substr(i + 1, s.size() - 1 - (i + 1));
    if (body.find_first_of("{}'") != string_view::npos) return fallback(R, "concatenation or literal");
    Inst in;
    in.name = canon(name_raw);
    in.master = string(master);
    in.escaped = name_raw[0] == '\\';
    if (!inst_seen.insert(in.name).second) return fallback(R, "duplicate instance");
    auto cit = L.cell_id.find(in.master);
    in.cell = cit == L.cell_id.end() ? -1 : cit->second;
    in.c0 = (i64)conns.size();
    inst_pins.clear();
    size_t p = 0;
    while (true) {
      while (p < body.size() && (ws(body[p]) || body[p] == ',')) p++;
      if (p >= body.size()) break;
      if (body[p] != '.') return fallback(R, "connection syntax");
      size_t q0 = ++p;
      if (p >= body.size() || !alpha_(body[p])) return fallback(R, "connection pin");
      while (p < body.size() && word(body[p])) p++;
      string_view pin = body.substr(q0, p - q0);
      while (p < body.size() && ws(body[p])) p++;
      if (p >= body.size() || body[p] != '(') return fallback(R, "connection syntax");
      size_t c0 = ++p;
      while (p < body.size() && body[p] != ')' && body[p] != '(') p++;
      if (p >= body.size() || body[p] != ')') return fallback(R, "connection syntax");
      string net = canon(body.substr(c0, p - c0));
      p++;
      if (net.empty()) return fallback(R, "unconnected pin");
      for (auto& q : inst_pins)
        if (q == pin) return fallback(R, "pin connected twice");
      inst_pins.push_back(pin);
      int local = -1;
      if (in.cell >= 0) {
        for (i64 g = L.pin_ptr[in.cell]; g < L.pin_ptr[in.cell + 1]; g++)
          if (L.pin_name[g] == pin) { local = (int)(g - L.pin_ptr[in.cell]); break; }
        if (local < 0) return fallback(R, "pin not in Liberty");
      }
      auto ins = conn_net_id.emplace(net, (int)conn_net_name.size());
      if (ins.second) conn_net_name.push_back(net);
      conns.push_back({local, ins.first->second});
    }
    in.c1 = (i64)conns.size();
    insts.push_back(std::move(in));
  }
  if (!have_top) return fallback(R, "no module");
  tick("parse statements");
  for (auto& h : header)
    if (!declared_bases.count(h)) return fallback(R, "header port not declared");

  {
    vector<string> bad;
    std::unordered_set<string> seen;
    for (auto& in : insts) {
      if (!seen.insert(in.master).second) continue;
      if (in.cell < 0 || !(L.cell_flags[in.cell] & 1)) bad.push_back(in.master);
    }
    R->n_masters = (i64)seen.size();
    if (!bad.empty()) {
      std::sort(bad.begin(), bad.end());
      R->status = FP_UNSUPPORTED;
      for (auto& m : bad) { R->message += m; R->message += '\0'; }
      return R;
    }
  }
  if (sdc->top && top != sdc->top) return fallback(R, "SDC current_design != top");
  for (auto& in : insts) {
    if (L.cell_flags[in.cell] & 4) return fallback(R, "broken Liberty cell");
    i64 np = L.pin_ptr[in.cell + 1] - L.pin_ptr[in.cell];
    if (in.c1 - in.c0 != np) return fallback(R, "unconnected Liberty pins");
  }

  std::unordered_map<string, int> net_id;
  vector<string> net_name;
  auto intern = [&](const string& s) {
    auto ins = net_id.emplace(s, (int)net_name.size());
    if (ins.second) net_name.push_back(s);
    return ins.first->second;
  };
  vector<int> port_net(ports.size());
  {
    std::unordered_set<string> pn;
    for (size_t i = 0; i < ports.size(); i++) {
      if (!pn.insert(ports[i].name).second) return fallback(R, "duplicate port");
      port_net[i] = intern(ports[i].name);
    }
    std::unordered_map<string_view, int, Hash> inst_of;
    for (size_t i = 0; i < insts.size(); i++) inst_of.emplace(insts[i].name, (int)i);
    for (auto& pt : ports) {
      size_t sl = pt.name.find('/');
      while (sl != string::npos) {
        auto it = inst_of.find(string_view(pt.name).substr(0, sl));
        if (it != inst_of.end() && L.pin_index[insts[it->second].cell].count(pt.name.substr(sl + 1)))
          return fallback(R, "port name collides with a pin name");
        sl = pt.name.find('/', sl + 1);
      }
    }
  }
  vector<int> remap(conn_net_name.size());
  for (size_t k = 0; k < conn_net_name.size(); k++) {
    const string& n = conn_net_name[k];
    if (n.find('[') == string::npos) {
      auto it = bus_range.find(n);
      if (it != bus_range.end()) {
        if (it->second.first != it->second.second) return fallback(R, "whole bus connected");
        remap[k] = intern(n + "[" + std::to_string(it->second.first) + "]");
        continue;
      }
    }
    remap[k] = intern(n);
  }
  for (auto& c : conns) c.net = remap[c.net];
  tick("intern nets");
  conn_net_id.clear();
  conn_net_name.clear();

  const i64 n_inst = (i64)insts.size(), n_ports = (i64)ports.size();
  vector<i64> first(n_inst);
  i64 N = n_ports;
  for (i64 i = 0; i < n_inst; i++) { first[i] = N; N += L.pin_ptr[insts[i].cell + 1] - L.pin_ptr[insts[i].cell]; }
  if (N == 0) return fallback(R, "empty design");
  R->N = N; R->n_ports = n_ports; R->n_inst = n_inst; R->top = top;
  vector<int8_t> role(N);
  vector<i64> master_id(N), pinname_id(N), pin_inst(N);
  vector<double> cap_rise(N), cap_fall(N), max_tr(N), max_cap(N);
  vector<int8_t> pin_is_input(N), pin_is_output(N);
  vector<int> node_cell_pin(N, -1);
  {
    size_t blob = 0;
    for (auto& pt : ports) blob += pt.name.size() + 1;
    for (auto& in : insts) blob += (in.name.size() + 2) * (L.pin_ptr[in.cell + 1] - L.pin_ptr[in.cell]) + 8 * 8;
    R->pin_blob.reserve(blob);
  }
  for (i64 i = 0; i < n_ports; i++) {
    role[i] = ports[i].dir == 0 ? 0 : 1;
    pin_inst[i] = n_inst + i;
    R->pin_blob += ports[i].name;
    R->pin_blob += '\0';
  }
  for (i64 ii = 0; ii < n_inst; ii++) {
    const Inst& in = insts[ii];
    R->inst_blob += in.name; R->inst_blob += '\0';
    R->master_blob += in.master; R->master_blob += '\0';
    R->escaped += in.escaped;
    if (L.cell_flags[in.cell] & 2) R->n_registers++;
    if (L.cell_flags[in.cell] & 8) R->tie_cells++;
    for (i64 g = L.pin_ptr[in.cell], v = first[ii]; g < L.pin_ptr[in.cell + 1]; g++, v++) {
      role[v] = L.pin_role[g];
      master_id[v] = in.cell;
      pinname_id[v] = L.pin_vocab[g];
      pin_inst[v] = ii;
      cap_rise[v] = L.pin_cap_rise[g];
      cap_fall[v] = L.pin_cap_fall[g];
      max_tr[v] = L.pin_max_tr[g];
      max_cap[v] = L.pin_max_cap[g];
      pin_is_input[v] = L.pin_dir[g] == 0;
      pin_is_output[v] = L.pin_dir[g] == 1;
      node_cell_pin[v] = (int)g;
      R->pin_blob += in.name; R->pin_blob += '/'; R->pin_blob += L.pin_name[g]; R->pin_blob += '\0';
    }
  }

  tick("nodes");
  const i64 n_nets = (i64)net_name.size();
  vector<i64> drv(n_nets, -1);
  vector<i64> driver_order;
  vector<std::pair<int, i64>> sink_app;
  sink_app.reserve(N);
  for (i64 i = 0; i < n_ports; i++) {
    if (ports[i].dir == 0) { drv[port_net[i]] = i; driver_order.push_back(port_net[i]); }
    else sink_app.push_back({port_net[i], i});
  }
  for (i64 ii = 0; ii < n_inst; ii++) {
    const Inst& in = insts[ii];
    for (i64 k = in.c0; k < in.c1; k++) {
      i64 v = first[ii] + conns[k].pin;
      int net = conns[k].net;
      if (pin_is_output[v]) {
        if (drv[net] >= 0) return fallback(R, "net with two drivers");
        drv[net] = v;
        driver_order.push_back(net);
      } else {
        sink_app.push_back({net, v});
      }
    }
  }
  vector<i64> sptr(n_nets + 1, 0), snode(sink_app.size());
  for (auto& sa : sink_app) sptr[sa.first + 1]++;
  for (i64 n = 0; n < n_nets; n++) sptr[n + 1] += sptr[n];
  {
    vector<i64> fill(sptr.begin(), sptr.end() - 1);
    for (auto& sa : sink_app) snode[fill[sa.first]++] = sa.second;
  }
  sink_app.clear();
  sink_app.shrink_to_fit();
  for (i64 n = 0; n < n_nets; n++)
    if (sptr[n + 1] > sptr[n] && drv[n] < 0) return fallback(R, "net without driver");
  R->nets = (i64)driver_order.size();
  vector<i64> drv_of_sink(N, -1);
  for (i64 n : driver_order) {
    i64 d = drv[n];
    if (sptr[n + 1] == sptr[n]) R->unloaded++;
    bool p2p = false;
    for (i64 k = sptr[n]; k < sptr[n + 1]; k++) {
      R->net_src.push_back(d);
      R->net_dst.push_back(snode[k]);
      drv_of_sink[snode[k]] = d;
      if (role[snode[k]] == 1) p2p = true;
    }
    if (role[d] == 0 && p2p) R->port_to_port++;
  }

  tick("nets");
  vector<double> load_rise(N, 0.0), load_fall(N, 0.0), fanout(N, 0.0);
  {
    vector<double> buf;
    for (i64 n : driver_order) {
      i64 d = drv[n], m = sptr[n + 1] - sptr[n];
      buf.resize(m);
      for (i64 k = 0; k < m; k++) buf[k] = cap_rise[snode[sptr[n] + k]];
      load_rise[d] = np_sum(buf.data(), m);
      for (i64 k = 0; k < m; k++) buf[k] = cap_fall[snode[sptr[n] + k]];
      load_fall[d] = np_sum(buf.data(), m);
      fanout[d] = (double)m;
    }
  }

  vector<double> in_delay(N, 0.0), out_delay(N, 0.0), is_clock_port(N, 0.0);
  {
    std::unordered_map<string_view, i64, Hash> port_node;
    for (i64 i = 0; i < n_ports; i++) port_node.emplace(ports[i].name, i);
    i64 clk = -1;
    if (sdc->clock_port) {
      auto it = port_node.find(sdc->clock_port);
      if (it == port_node.end() || ports[it->second].dir != 0) return fallback(R, "SDC clock port");
      clk = it->second;
      is_clock_port[clk] = 1.0;
    }
    vector<int8_t> has_in(n_ports, 0), has_out(n_ports, 0);
    const char* p = sdc->in_names;
    for (i64 k = 0; k < sdc->n_in; k++, p += strlen(p) + 1) {
      auto it = port_node.find(p);
      if (it == port_node.end() || ports[it->second].dir != 0) return fallback(R, "SDC input delay port");
      in_delay[it->second] = sdc->in_vals[k];
      has_in[it->second] = 1;
    }
    p = sdc->out_names;
    for (i64 k = 0; k < sdc->n_out; k++, p += strlen(p) + 1) {
      auto it = port_node.find(p);
      if (it == port_node.end() || ports[it->second].dir != 1) return fallback(R, "SDC output delay port");
      out_delay[it->second] = sdc->out_vals[k];
      has_out[it->second] = 1;
    }
    for (i64 i = 0; i < n_ports; i++) {
      if (ports[i].dir == 0 && i != clk && !has_in[i]) {
        return fallback(R, "input port without delay");
      }
      if (ports[i].dir == 1 && !has_out[i]) return fallback(R, "output port without delay");
    }
  }

  tick("loads + SDC");
  vector<int8_t> cval(N, 2);
  for (i64 ii = 0; ii < n_inst; ii++)
    for (i64 g = L.pin_ptr[insts[ii].cell], v = first[ii]; g < L.pin_ptr[insts[ii].cell + 1]; g++, v++)
      if (L.pin_const_init[g] >= 0) cval[v] = L.pin_const_init[g];
  vector<int8_t> partial_flag(n_inst, 0);
  vector<int8_t> pinval(64), slotval(128);
  auto load_env = [&](i64 ii) {
    const Inst& in = insts[ii];
    i64 g0 = L.pin_ptr[in.cell], np = L.pin_ptr[in.cell + 1] - g0;
    if ((i64)pinval.size() < np) pinval.resize(np);
    bool any = false;
    for (i64 k = 0; k < np; k++) {
      int8_t v = 2;
      if (L.pin_dir[g0 + k] == 0 && cval[first[ii] + k] != 2) { v = cval[first[ii] + k]; any = true; }
      pinval[k] = v;
    }
    return any;
  };
  {
    bool any_const = false;
    for (i64 v = 0; v < N && !any_const; v++) any_const = cval[v] != 2;
    bool changed = any_const;
    while (changed) {
      changed = false;
      for (i64 n : driver_order) {
        i64 d = drv[n];
        if (cval[d] != 2)
          for (i64 k = sptr[n]; k < sptr[n + 1]; k++)
            if (cval[snode[k]] == 2) { cval[snode[k]] = cval[d]; changed = true; }
      }
      for (i64 ii = 0; ii < n_inst; ii++) {
        if (!load_env(ii)) continue;
        const Inst& in = insts[ii];
        i64 g0 = L.pin_ptr[in.cell];
        for (i64 g = g0; g < L.pin_ptr[in.cell + 1]; g++) {
          if (L.pin_dir[g] != 1 || L.pin_func[g] == -1) continue;
          i64 out = first[ii] + (g - g0);
          if (cval[out] != 2) continue;
          int val = 2;
          if (L.pin_func[g] >= 0) {
            fill_slots(L, L.pin_func[g], pinval.data(), slotval.data());
            val = eval_expr(L, L.pin_func[g], slotval.data());
          }
          if (val != 2) { cval[out] = (int8_t)val; changed = true; }
          else partial_flag[ii] = 1;
        }
      }
    }
  }
  vector<double> is_constant(N, 0.0);
  for (i64 v = 0; v < N; v++) if (cval[v] != 2) { is_constant[v] = 1.0; R->n_const++; }
  for (i64 ii = 0; ii < n_inst; ii++) {
    if (!partial_flag[ii]) continue;
    bool all = true;
    i64 g0 = L.pin_ptr[insts[ii].cell];
    for (i64 g = g0; g < L.pin_ptr[insts[ii].cell + 1]; g++)
      if (L.pin_dir[g] == 1 && cval[first[ii] + (g - g0)] == 2) all = false;
    if (!all) R->partial.push_back(ii);
  }

  tick("constants");
  {
    i64 E = 0;
    for (i64 ii = 0; ii < n_inst; ii++) E += L.arc_ptr[insts[ii].cell + 1] - L.arc_ptr[insts[ii].cell];
    for (auto* v : {&R->a_src, &R->a_dst, &R->a_type, &R->a_sense, &R->a_when, &R->a_key, &R->a_inst}) v->reserve(E);
    R->a_rf.reserve(4 * E); R->a_when_present.reserve(E); R->a_enabled.reserve(E);
  }
  vector<int> pm_count(L.n_cells, 0);
  vector<std::pair<i64, i64>> true_pairs;
  vector<int8_t> base(128);
  for (i64 ii = 0; ii < n_inst; ii++) {
    const Inst& in = insts[ii];
    const i64 cell = in.cell, g0 = L.pin_ptr[cell];
    bool env = load_env(ii);
    true_pairs.clear();
    if (env) {
      for (i64 a = L.arc_ptr[cell]; a < L.arc_ptr[cell + 1]; a++) {
        i64 e = L.arc_when_expr[a];
        if (e < 0) continue;
        fill_slots(L, e, pinval.data(), slotval.data());
        if (eval_expr(L, e, slotval.data()) == 1) true_pairs.push_back({L.arc_from[a], L.arc_to[a]});
      }
    }
    for (i64 a = L.arc_ptr[cell]; a < L.arc_ptr[cell + 1]; a++) {
      i64 f = L.arc_from[a], t = L.arc_to[a];
      i64 u = first[ii] + f, o = first[ii] + t;
      R->a_src.push_back(u); R->a_dst.push_back(o);
      R->a_type.push_back(L.arc_type[a]); R->a_sense.push_back(L.arc_sense[a]);
      for (int k = 0; k < 4; k++) R->a_rf.push_back((uint8_t)L.arc_rf[4 * a + k]);
      R->a_when.push_back(L.arc_when_vocab[a]);
      R->a_when_present.push_back(L.arc_when_vocab[a] != 0);
      R->a_key.push_back(L.arc_key[a]);
      R->a_inst.push_back(ii);
      bool enabled = cval[u] == 2 && cval[o] == 2;
      i64 we = L.arc_when_expr[a];
      if (enabled && we != -1 && env && we >= 0) {
        fill_slots(L, we, pinval.data(), slotval.data());
        if (eval_expr(L, we, slotval.data()) == 0) { enabled = false; R->when_disabled++; }
      }
      if (enabled && we == -1) {
        for (auto& pr : true_pairs)
          if (pr.first == f && pr.second == t) { enabled = false; R->default_disabled++; break; }
      }
      i64 fn = L.pin_func[g0 + t], slot = L.arc_collapse_slot[a];
      if (enabled && env && fn >= 0 && L.pin_dir[g0 + f] == 0 && slot >= 0) {
        fill_slots(L, fn, pinval.data(), base.data());
        if (!depends_on(L, fn, slot, base.data())) { enabled = false; R->collapsed++; }
      }
      R->a_enabled.push_back(enabled);
      if (pm_count[cell]++ == 0) R->pm_master.push_back(cell);
    }
  }
  for (i64 m : R->pm_master) R->pm_count.push_back(pm_count[m]);

  for (i64 ii = 0; ii < n_inst; ii++) {
    i64 cell = insts[ii].cell;
    if (!(L.cell_flags[cell] & 2)) continue;
    for (i64 c = L.chk_ptr[cell]; c < L.chk_ptr[cell + 1]; c++) {
      R->c_src.push_back(first[ii] + L.chk_clk[c]);
      R->c_dst.push_back(first[ii] + L.chk_d[c]);
      R->c_bank.push_back(L.chk_bank[c]);
    }
  }

  tick("arc + check edges");
  {
    const i64 En = (i64)R->net_src.size(), Ea = (i64)R->a_src.size();
    vector<i64> ptr(N + 1, 0), adj(En + Ea), indeg(N, 0);
    for (i64 k = 0; k < En; k++) { ptr[R->net_src[k] + 1]++; indeg[R->net_dst[k]]++; }
    for (i64 k = 0; k < Ea; k++) { ptr[R->a_src[k] + 1]++; indeg[R->a_dst[k]]++; }
    for (i64 v = 0; v < N; v++) ptr[v + 1] += ptr[v];
    vector<i64> fill(ptr.begin(), ptr.end() - 1);
    for (i64 k = 0; k < En; k++) adj[fill[R->net_src[k]]++] = R->net_dst[k];
    for (i64 k = 0; k < Ea; k++) adj[fill[R->a_src[k]]++] = R->a_dst[k];
    R->level.assign(N, 0);
    vector<i64> q;
    q.reserve(N);
    for (i64 v = 0; v < N; v++) if (indeg[v] == 0) q.push_back(v);
    for (size_t h = 0; h < q.size(); h++) {
      i64 u = q[h];
      for (i64 k = ptr[u]; k < ptr[u + 1]; k++) {
        i64 v = adj[k];
        if (R->level[u] + 1 > R->level[v]) R->level[v] = R->level[u] + 1;
        if (--indeg[v] == 0) q.push_back(v);
      }
    }
    if ((i64)q.size() != N) return fallback(R, "combinational loop");
  }

  tick("levels");
  R->x_cat.resize(3 * N); R->x_num.resize(9 * N); R->x_erc.resize(2 * N);
  for (i64 v = 0; v < N; v++) {
    R->x_cat[3 * v] = role[v]; R->x_cat[3 * v + 1] = master_id[v]; R->x_cat[3 * v + 2] = pinname_id[v];
    float* x = &R->x_num[9 * v];
    x[0] = (float)cap_rise[v]; x[1] = (float)cap_fall[v]; x[2] = (float)load_rise[v]; x[3] = (float)load_fall[v];
    x[4] = (float)fanout[v]; x[5] = (float)in_delay[v]; x[6] = (float)out_delay[v]; x[7] = (float)is_constant[v];
    x[8] = (float)is_clock_port[v];
    R->x_erc[2 * v] = (float)max_tr[v]; R->x_erc[2 * v + 1] = (float)max_cap[v];
  }
  R->pin_instance = std::move(pin_inst);

  if (label_dir) {
    string dir(label_dir), ep_text, pt_text;
    tick("node tensors");
    if (!read_file((dir + "/endpoints.tsv").c_str(), ep_text) || !read_gz((dir + "/points.tsv.gz").c_str(), pt_text))
      return fallback(R, "cannot read labels");
    tick("labels: read + gunzip");
    if (!plain_ascii(ep_text) || !plain_ascii(pt_text) || ep_text.find('"') != string::npos)
      return fallback(R, "label files not plain ASCII TSV");
    auto num = [](string_view s, double& out) {
      if (s.empty() || s.find_first_not_of("0123456789.eE+-") != string_view::npos) return false;
      string tmp(s);
      char* end;
      out = strtod(tmp.c_str(), &end);
      return end == tmp.c_str() + tmp.size();
    };
    auto integer = [](string_view s, i64& out) {
      size_t i = 0;
      return parse_int(s, i, out) && i == s.size();
    };
    auto lines = [](string_view t, vector<string_view>& out) {
      out.clear();
      size_t a = 0;
      while (a < t.size()) {
        size_t b = t.find('\n', a);
        if (b == string_view::npos) b = t.size();
        out.push_back(t.substr(a, b - a));
        a = b + 1;
      }
    };
    auto fields = [](string_view l, vector<string_view>& out) {
      out.clear();
      size_t a = 0;
      while (true) {
        size_t b = l.find('\t', a);
        out.push_back(l.substr(a, b == string_view::npos ? string_view::npos : b - a));
        if (b == string_view::npos) break;
        a = b + 1;
      }
    };
    vector<string_view> L1, F;
    lines(pt_text, L1);
    struct Pt { string_view pin; int8_t rf; double arr; };
    vector<Pt> pts;
    pts.reserve(L1.size());
    std::unordered_map<i64, vector<int>> by_pid;
    for (size_t k = 1; k < L1.size(); k++) {
      fields(L1[k], F);
      i64 pid, idx;
      double arr;
      if (F.size() != 5 || !integer(F[0], pid) || !integer(F[1], idx) || !num(F[4], arr)) return fallback(R, "points line");
      by_pid[pid].push_back((int)pts.size());
      pts.push_back({F[2], (int8_t)(F[3] == "^" ? 0 : 1), arr});
    }
    if (L1.empty()) return fallback(R, "empty points file");
    tick("labels: parse points");
    vector<string_view> L2;
    lines(ep_text, L2);
    if (L2.empty()) return fallback(R, "empty endpoints file");
    fields(L2[0], F);
    int c_pid = -1, c_ep = -1, c_rank = -1, c_sl = -1, c_ar = -1, c_rq = -1;
    const size_t ncol = F.size();
    for (size_t c = 0; c < F.size(); c++) {
      if (F[c] == "path_id") c_pid = (int)c;
      else if (F[c] == "endpoint") c_ep = (int)c;
      else if (F[c] == "rank") c_rank = (int)c;
      else if (F[c] == "slack_ps") c_sl = (int)c;
      else if (F[c] == "arrival_ps") c_ar = (int)c;
      else if (F[c] == "required_ps") c_rq = (int)c;
    }
    if (std::min({c_pid, c_ep, c_rank, c_sl, c_ar, c_rq}) < 0) return fallback(R, "endpoints header");
    std::unordered_map<string_view, i64, Hash> node_of;
    node_of.reserve(N * 2);
    {
      size_t a = 0;
      for (i64 v = 0; v < N; v++) {
        size_t b = R->pin_blob.find('\0', a);
        node_of.emplace(string_view(R->pin_blob).substr(a, b - a), v);
        a = b + 1;
      }
    }
    vector<i64> aptr(N + 1, 0), adst(R->a_src.size());
    for (i64 u : R->a_src) aptr[u + 1]++;
    for (i64 v = 0; v < N; v++) aptr[v + 1] += aptr[v];
    {
      vector<i64> fill(aptr.begin(), aptr.end() - 1);
      for (size_t k = 0; k < R->a_src.size(); k++) adst[fill[R->a_src[k]]++] = R->a_dst[k];
    }
    auto is_arc = [&](i64 u, i64 v) {
      for (i64 k = aptr[u]; k < aptr[u + 1]; k++) if (adst[k] == v) return true;
      return false;
    };
    vector<i64> nodes;
    auto path_nodes = [&](i64 pid) -> bool {
      auto it = by_pid.find(pid);
      if (it == by_pid.end() || it->second.empty()) return false;
      nodes.clear();
      for (int k : it->second) {
        auto nt = node_of.find(pts[k].pin);
        if (nt == node_of.end()) return false;
        nodes.push_back(nt->second);
      }
      bool prev_net = false;
      for (size_t k = 0; k + 1 < nodes.size(); k++) {
        bool net = drv_of_sink[nodes[k + 1]] == nodes[k];
        if (!net && !is_arc(nodes[k], nodes[k + 1])) return false;
        if (net && prev_net && k > 0) return false;
        prev_net = net;
      }
      return true;
    };
    R->y_ep_ptr.push_back(0);
    vector<double> ep_slack_d;
    bool first_row = true;
    for (size_t k = 1; k < L2.size(); k++) {
      if (L2[k].empty()) continue;
      fields(L2[k], F);
      i64 rank, pid;
      double sl, ar, rq;
      if (F.size() != ncol || !integer(F[c_rank], rank)) return fallback(R, "endpoints line");
      if (rank != 0) continue;
      if (!integer(F[c_pid], pid) || !num(F[c_sl], sl) || !num(F[c_ar], ar) || !num(F[c_rq], rq)) return fallback(R, "endpoints line");
      if (!path_nodes(pid)) return fallback(R, "label path inconsistent with the graph");
      auto ep = node_of.find(F[c_ep]);
      if (ep == node_of.end() || ep->second != nodes.back()) return fallback(R, "endpoint does not end its path");
      int r0 = role[nodes.front()], r1 = role[nodes.back()];
      if (!(r0 == 0 || r0 == 6) || !(r1 == 1 || r1 == 5)) return fallback(R, "bad start/end role");
      R->y_ep_index.push_back(ep->second);
      ep_slack_d.push_back(sl);
      R->y_ep_slack.push_back((float)sl); R->y_ep_arr.push_back((float)ar); R->y_ep_req.push_back((float)rq);
      if (first_row || sl < R->worst_slack) R->worst_slack = sl;
      auto& ids = by_pid[pid];
      for (size_t j = 0; j < nodes.size(); j++) {
        R->y_ep_path.push_back(nodes[j]);
        R->y_ep_rf.push_back(pts[ids[j]].rf);
        R->y_ep_path_arr.push_back((float)pts[ids[j]].arr);
      }
      R->y_ep_ptr.push_back((i64)R->y_ep_path.size());
      if (first_row) {
        R->first_slack = sl;
        R->y_path = nodes;
        for (int id : ids) { R->y_path_rf.push_back(pts[id].rf); R->y_path_arr.push_back((float)pts[id].arr); }
        first_row = false;
      }
    }
    if (first_row) return fallback(R, "no rank-0 path");
    for (size_t e = 0; e < ep_slack_d.size(); e++) {
      double d = ep_slack_d[e] - R->worst_slack;
      if ((d < 0 ? -d : d) < 1e-3) R->y_worst.push_back(R->y_ep_index[e]);
    }
    R->has_labels = 1;
    tick("labels: paths");
  }
  R->status = FP_OK;
  return R;
}

template <class T>
static void put(void* dst, const vector<T>& v) { if (!v.empty()) memcpy(dst, v.data(), v.size() * sizeof(T)); }
static void put2(void* dst, const vector<i64>& a, const vector<i64>& b) {
  put(dst, a);
  if (!b.empty()) memcpy((i64*)dst + a.size(), b.data(), b.size() * sizeof(i64));
}

extern "C" {

void* fp_lib_new(const LibSpec* s) {
  Lib* L = new Lib();
  L->n_cells = s->n_cells;
  const char* p = s->cell_names;
  for (i64 c = 0; c < s->n_cells; c++, p += strlen(p) + 1) {
    L->cell_name.emplace_back(p);
    L->cell_id.emplace(p, (int)c);
  }
  L->cell_flags = cp(s->cell_flags, s->n_cells);
  L->pin_ptr = cp(s->pin_ptr, s->n_cells + 1);
  const i64 P = L->pin_ptr.back();
  p = s->pin_names;
  for (i64 g = 0; g < P; g++, p += strlen(p) + 1) L->pin_name.emplace_back(p);
  L->pin_index.resize(s->n_cells);
  for (i64 c = 0; c < s->n_cells; c++)
    for (i64 g = L->pin_ptr[c]; g < L->pin_ptr[c + 1]; g++) L->pin_index[c].emplace(L->pin_name[g], (int)(g - L->pin_ptr[c]));
  L->pin_vocab = cp(s->pin_vocab, P);
  L->pin_dir = cp(s->pin_dir, P);
  L->pin_role = cp(s->pin_role, P);
  L->pin_cap_rise = cp(s->pin_cap_rise, P);
  L->pin_cap_fall = cp(s->pin_cap_fall, P);
  L->pin_max_tr = cp(s->pin_max_tr, P);
  L->pin_max_cap = cp(s->pin_max_cap, P);
  L->pin_const_init = cp(s->pin_const_init, P);
  L->pin_func = cp(s->pin_func, P);
  L->arc_ptr = cp(s->arc_ptr, s->n_cells + 1);
  const i64 A = L->arc_ptr.back();
  L->arc_from = cp(s->arc_from, A);
  L->arc_to = cp(s->arc_to, A);
  L->arc_type = cp(s->arc_type, A);
  L->arc_sense = cp(s->arc_sense, A);
  L->arc_rf = cp(s->arc_rf, 4 * A);
  L->arc_when_vocab = cp(s->arc_when_vocab, A);
  L->arc_key = cp(s->arc_key, A);
  L->arc_when_expr = cp(s->arc_when_expr, A);
  L->arc_collapse_slot = cp(s->arc_collapse_slot, A);
  L->chk_ptr = cp(s->chk_ptr, s->n_cells + 1);
  const i64 C = L->chk_ptr.back();
  L->chk_clk = cp(s->chk_clk, C);
  L->chk_d = cp(s->chk_d, C);
  L->chk_bank = cp(s->chk_bank, C);
  L->expr_ptr = cp(s->expr_ptr, s->n_expr + 1);
  L->expr_code = cp(s->expr_code, L->expr_ptr.back());
  L->expr_slot_ptr = cp(s->expr_slot_ptr, s->n_expr + 1);
  L->expr_slot_pin = cp(s->expr_slot_pin, L->expr_slot_ptr.back());
  return L;
}

void fp_lib_free(void* l) { delete (Lib*)l; }

void* fp_build(void* lib, const char* netlist, const SdcSpec* sdc, const char* label_dir) {
  try {
    return build(*(Lib*)lib, netlist, sdc, label_dir);
  } catch (const std::exception& e) {
    Result* r = new Result();
    return fallback(r, string("C++ exception: ") + e.what());
  }
}

int fp_status(void* r) { return ((Result*)r)->status; }
i64 fp_message(void* r, char* buf, i64 cap) {
  auto& m = ((Result*)r)->message;
  if (buf) memcpy(buf, m.data(), std::min<i64>(cap, (i64)m.size()));
  return (i64)m.size();
}

void fp_sizes(void* rr, i64* s) {
  Result* r = (Result*)rr;
  i64 v[] = {r->N, r->n_ports, r->n_inst, (i64)r->net_src.size(), (i64)r->a_src.size(), (i64)r->c_src.size(),
             (i64)r->pin_blob.size(), (i64)r->inst_blob.size(), (i64)r->master_blob.size(), (i64)r->top.size(),
             (i64)r->partial.size(), (i64)r->pm_master.size(), r->has_labels, (i64)r->y_path.size(),
             (i64)r->y_worst.size(), (i64)r->y_ep_index.size(), (i64)r->y_ep_path.size(),
             r->nets, r->unloaded, r->port_to_port, r->when_disabled, r->default_disabled, r->collapsed,
             r->escaped, r->n_masters, r->n_registers, r->tie_cells, r->n_const};
  memcpy(s, v, sizeof v);
}

void fp_fill(void* rr, void** out) {
  Result* r = (Result*)rr;
  put(out[0], r->x_cat); put(out[1], r->x_num); put(out[2], r->x_erc); put(out[3], r->pin_instance); put(out[4], r->level);
  put2(out[5], r->net_src, r->net_dst); put2(out[6], r->a_src, r->a_dst);
  put(out[7], r->a_type); put(out[8], r->a_sense); put(out[9], r->a_rf); put(out[10], r->a_when_present);
  put(out[11], r->a_when); put(out[12], r->a_key); put(out[13], r->a_inst); put(out[14], r->a_enabled);
  put2(out[15], r->c_src, r->c_dst); put(out[16], r->c_bank);
  memcpy(out[17], r->pin_blob.data(), r->pin_blob.size()); memcpy(out[18], r->inst_blob.data(), r->inst_blob.size());
  memcpy(out[19], r->master_blob.data(), r->master_blob.size()); memcpy(out[20], r->top.data(), r->top.size());
  put(out[21], r->partial); put(out[22], r->pm_master); put(out[23], r->pm_count);
  if (r->has_labels) {
    put(out[24], r->y_path); put(out[25], r->y_path_rf); put(out[26], r->y_path_arr); put(out[27], r->y_worst);
    put(out[28], r->y_ep_index); put(out[29], r->y_ep_slack); put(out[30], r->y_ep_arr); put(out[31], r->y_ep_req);
    put(out[32], r->y_ep_path); put(out[33], r->y_ep_ptr); put(out[34], r->y_ep_rf); put(out[35], r->y_ep_path_arr);
    double d[2] = {r->worst_slack, r->first_slack};
    memcpy(out[36], d, sizeof d);
  }
}

void fp_free(void* r) { delete (Result*)r; }

double fp_np_sum(const double* a, i64 n) { return np_sum(a, n); }
}
