---
modality: image
---
# Content Credentials (C2PA) and provenance metadata

Content Credentials (the C2PA standard) embed a signed manifest in an image file. The manifest lists who created the file, with which software, and what actions were applied. The IPTC digital source type `trainedAlgorithmicMedia` is the standard label for media created by a generative model; several AI generators (for example Microsoft Bing Image Creator and other major image services) write it automatically.

## How to read it
- A manifest that declares `trainedAlgorithmicMedia` is a strong statement of AI origin from the tool that made the file.
- Validation matters: a "Trusted" state means the signing chain is on the verifier's trust list. "Invalid" or "untrusted" often means the issuer is simply not on the local trust list, not that the file was forged, but it means the declaration is unauthenticated.
- A pixel-hash mismatch means the image was modified after it was signed.

## Limits
Metadata is easily stripped (screenshots, social-media re-uploads, format conversion). The absence of Content Credentials says nothing about authenticity. Presence of a credential on a real photograph is also possible if the photo was edited with a Content-Credentials-aware tool.
