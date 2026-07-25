# Chrome Web Store — submission pack

Everything the developer dashboard asks for. Paste each field as-is.

---

## Name (45 char limit)

```
PhishSentry — Lookalike Domain Checker
```

## Short description (132 char limit)

```
Warns you when a site's domain imitates a known brand: character swaps, homograph spoofing, brand names on unrelated domains.
```

## Category

`Productivity` → sub-category `Workflow & Planning`
(Chrome has no "Security" consumer category; Productivity is the accepted home
for this kind of tool.)

---

## Detailed description

```
PhishSentry checks one specific thing, and tells you exactly what it found.

When you land on a site, it examines the domain name for the patterns attackers
use to imitate brands you trust:

• Character substitution — paypa1.com instead of paypal.com, g00gle.net, netf1ix.com
• Brand names on domains the brand does not own — paypal-secure.com, fedexverify.com, login-steam.com
• Homograph and internationalised spoofing — Cyrillic and Greek characters that render as Latin ones
• Brand names in a subdomain of an unrelated site — paypal.account-services.net
• Typosquats — a character doubled, dropped, or swapped: ppaypal.com, payapl.com, amazn.com
• Brand names on low-reputation top-level domains

Every warning states the reason. Instead of a confidence score you cannot
interrogate, you get "'paypal' appears in the domain 'paypa1-secure.com', which
is not owned by paypal." You can judge the reasoning yourself.

WHAT IT WILL NOT DO

This is a domain-similarity checker, and the listing is deliberately specific
about that:

• It does NOT detect phishing hosted on a legitimate platform under an unrelated
  subdomain — for example a credential form at a random subdomain of a free
  hosting service. Those domains are not imitations, and the domain name carries
  no evidence of the page's contents.
• It does NOT scan page content, attachments, or downloads.
• It is NOT a replacement for Google Safe Browsing, antivirus software, or your
  own judgement.

Measured performance and every known limitation are published in the repository
(EVALUATION.md), including the measurements that led to this scope.

PRIVACY

• Only the HOSTNAME is ever sent — never the path, query string, or page content.
• A built-in list of widely-used domains is checked entirely inside your browser.
  Those hostnames are never sent anywhere at all, so most ordinary browsing never
  leaves your device.
• The checking service is stateless. Hostnames are not stored and are excluded
  from its logs.
• No accounts. No analytics. No third parties. No data is sold.

OPEN SOURCE

Extension and checking service:
https://github.com/abhishekjaden/Phish-Sentry

Detection is deterministic rules, not a machine-learning model, which is why
each verdict can name its reason — and why known-good brand domains are
allowlisted by construction rather than merely scoring low.
```

---

## Single purpose description

(Dashboard field: "Single purpose")

```
The extension has one purpose: to warn the user when the domain of the page they
are viewing imitates a well-known brand. It compares the hostname against a list
of known brand domains using character-substitution, homograph, edit-distance,
and subdomain-placement rules, then reports whether an imitation pattern matched
and which rule matched. It performs no other function.
```

---

## Permission justifications

Paste each into its matching dashboard field.

**`tabs`**
```
Needed to read the address of the active tab so its domain can be checked for
brand imitation. Only the hostname is used; the path and query string are
discarded and never transmitted. Without this permission the extension cannot
know which domain to check.
```

**`storage`**
```
Used to cache verdicts locally so that revisiting a site does not require a
repeated network request. The cache holds at most 300 hostname/verdict pairs,
each expiring after 30 minutes, and never leaves the user's device.
```

**`notifications`**
```
Used to show a desktop notification when the current domain appears to imitate a
known brand, so the user is warned before entering credentials on a page they may
already be interacting with.
```

**Host permission — `https://<api-id>.execute-api.ap-south-1.amazonaws.com/*`**
```
This is the extension's own checking endpoint, which receives a hostname and
returns whether it matches a brand-imitation pattern. It is the only remote host
the extension contacts.
```

**Content script on `http://*/*` and `https://*/*`**
```
The content script exists solely to draw a dismissible warning banner inside a
page whose domain matches an imitation pattern. Because an imitation domain can
be any domain, the script must be able to run on any page in order to display
that warning. It does not read page content, form fields, cookies, or user input;
it only appends a banner element and removes it when dismissed. Its full source
is content.js in the public repository.
```

---

## Data usage disclosures

Answer the dashboard checkboxes exactly as follows. These are the honest
answers — misdeclaring data handling is a common cause of takedown.

| Category | Collected? | Note |
|---|---|---|
| Personally identifiable information | **No** | |
| Health information | **No** | |
| Financial and payment information | **No** | |
| Authentication information | **No** | No accounts; no credentials handled |
| Personal communications | **No** | |
| Location | **No** | |
| **Web history** | **YES** | See below |
| User activity | **No** | No clicks, scrolls, or keystrokes recorded |
| Website content | **No** | Page content is never read or transmitted |

**Why "Web history" must be YES.** Chrome defines "collect" as transmitting data
off the user's device. The extension transmits the hostname of sites the user
visits to its checking endpoint. Even though nothing is stored and paths are
never sent, hostnames of visited sites fall under web history and must be
declared. Do not answer No here.

If a free-text field is offered alongside it:
```
Hostnames of visited sites are transmitted to the extension's own endpoint in
order to check them for brand-imitation patterns. Only the hostname is sent —
never the path, query string, or page content. The endpoint is stateless: it
stores nothing and excludes hostnames from its logs. A built-in list of
widely-used domains is resolved locally and those hostnames are never
transmitted at all.
```

**Required certifications** — all three can be truthfully affirmed:
- Not selling or transferring user data to third parties ✓
- Not using or transferring data for purposes unrelated to the single purpose ✓
- Not using or transferring data to determine creditworthiness or for lending ✓

**Privacy policy URL** — required. Must be a stable, always-available URL.
Do **not** use phishsentry.app: that resolves only while the EKS cluster is
running, and a dead privacy-policy link is grounds for rejection or removal.
Publish PRIVACY.md via GitHub Pages instead, for example:
`https://abhishekjaden.github.io/Phish-Sentry/privacy.html`

---

## Screenshots (1280×800 or 640×400, at least one, up to five)

1. Popup on a flagged domain — red state, showing the reason text and rule tag.
   This is the strongest shot: it shows the tool explaining itself.
2. Popup on a locally-verified domain (github.com) — green, with the
   "not sent anywhere" line visible. Demonstrates the privacy property.
3. Popup in the amber caution state on a free-hosting subdomain, showing the
   honest ambiguity wording.
4. The in-page warning banner.

Avoid: any real inbox, personal tabs, bookmarks bar, or third-party PII.

---

## Pre-submission checklist

- [ ] Privacy policy published at a stable URL (GitHub Pages, not phishsentry.app)
- [ ] Manifest `host_permissions` contains only the API endpoint — no `<all_urls>`
- [ ] Version is 2.0.0 and login code is fully removed
- [ ] Endpoint tested from a clean profile with the extension freshly installed
- [ ] Screenshots contain no personal data
- [ ] EVALUATION.md updated with the URL-model and lookalike measurements, since
      the listing points at it
- [ ] $5 one-time developer registration fee paid
