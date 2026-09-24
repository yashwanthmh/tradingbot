"""The research side: trials, the sealed holdout, and the selection statistics.

Separate from `registry` because the trust boundary runs between them. This
package generates and measures candidates; `registry` decides whether one may
have money. Nothing here can promote anything, and nothing here holds a broker
credential — the research process is expected to run without the live key in
its environment at all.
"""
