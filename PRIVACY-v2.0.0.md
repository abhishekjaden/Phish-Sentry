# Privacy Policy — PhishSentry Lookalike Domain Checker

**Last updated:** 25 July 2026
**Extension version:** 2.0.0

## Summary

The extension checks whether the domain you are visiting imitates a known
brand. To do that it sends **the hostname only** to a checking service. It does
not send the page path, query string, or page content. Nothing is stored on the
server. There are no accounts, no analytics, and no third parties.

## What is transmitted

When you visit a page, or press **Check this page**, the extension may send a
single piece of information to its checking endpoint:

* **The hostname of the site**, for example `example-login.com`.

It does **not** transmit:

* the full URL, the path, the query string, or the URL fragment
* page content, form fields, cookies, or anything you type
* your IP address beyond what any ordinary HTTPS request necessarily reveals to
  the receiving server
* any identifier for you, your browser, or your installation

Requests carry no user ID, no device ID, and no session token. Requests from
different users are indistinguishable from one another apart from network-level
metadata.

### Domains that are never transmitted

The extension ships with a built-in list of widely-used domains (search
engines, major cloud and developer platforms, banks, universities, news sites,
and similar). Pages on those domains are resolved **entirely inside your
browser** and their hostnames are **never sent anywhere at all**. In ordinary
browsing this means most of the sites you visit never leave your device.

You can read the exact list in `background.js` in the public source repository.

## What is stored

**On the server: nothing.** The checking service is stateless. It holds no
database and writes no records of the hostnames it receives. Its logs record
only the verdict (whether a lookalike pattern matched) and which rule matched.
Hostnames are deliberately excluded from those logs.

**In your browser:** the extension caches verdicts for hostnames you have
visited, in Chrome's local extension storage, so repeated visits do not need a
repeated request. This cache holds at most 300 entries, expires after 30
minutes per entry, never leaves your device, and is deleted when you remove the
extension. You can clear it at any time by removing and reinstalling the
extension.

## What is not done with your data

* Data is **not sold** to anyone, and is not transferred to any third party.
* Data is **not used** for advertising, profiling, tracking, or any purpose
  other than answering the single question "does this domain imitate a known
  brand?"
* Data is **not used** to determine creditworthiness or for lending decisions.
* There is **no analytics SDK**, no telemetry, and no error-reporting service.

## Permissions, and why each is needed

* **`tabs`** — to read the address of the tab you are viewing, so its domain can
  be checked. Used for the domain only.
* **`storage`** — to keep the local verdict cache described above.
* **`notifications`** — to show a desktop notification when a domain appears to
  imitate a brand.
* **Content script on all pages** — to display a dismissible warning banner
  inside a page whose domain looks like an imitation. The script only draws that
  banner. It does not read page content, form data, or anything you type.
* **Host access to the checking endpoint** — the extension can make requests to
  its own checking endpoint and to no other server.

## What this extension does not do

It checks domain names for imitation patterns. It does **not** detect phishing
hosted on a legitimate platform under an unrelated subdomain (for example, a
credential form hosted at a random subdomain of a free hosting provider), and it
does not inspect page content. It is not a replacement for Google Safe Browsing,
antivirus software, or your own judgement. Measured performance figures and
known limitations are published in `EVALUATION.md` in the source repository.

## Source code

The extension and the checking service are open source and can be audited:
https://github.com/abhishekjaden/Phish-Sentry

If this policy and the code ever disagree, the code is the authority — please
open an issue and the policy will be corrected.

## Changes to this policy

Material changes will be accompanied by a version increment here and in the
extension. The extension will not begin collecting a new category of data
without this policy being updated first.

## Contact

Abhishek Jaden — via the GitHub repository above.
