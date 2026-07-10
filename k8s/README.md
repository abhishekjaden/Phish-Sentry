# PhishSentry — Kubernetes (EKS) manifests

Deploys the full stack to EKS. The container images (app, inference, rag) must
be built and pushed to ECR first; the manifests reference
`<ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/phishsentry-*:latest`.

## Prerequisites (in order)
1. **Cluster up:** `eksctl create cluster -f ../eksctl-cluster.yaml`
2. **AWS Load Balancer Controller** installed via Helm (uses the
   `aws-load-balancer-controller` IRSA service account from the eksctl config) —
   required for the Ingress to provision an ALB.
3. **metrics-server** installed — required for the HPA to read CPU.
4. **ACM certificate** for `phishsentry.app` issued in `ap-south-1`,
   DNS-validated via a TXT record in Cloudflare. Put its ARN in `09-ingress.yaml`.
5. **Images pushed to ECR**; replace `<ACCOUNT_ID>` in 05/06/07/08.
6. **Real Secret created** on the cluster (do NOT apply the example):
   see the command block in `01-secrets.example.yaml`.

## Apply order
`kubectl apply -f .` respects filename ordering, but explicitly:

```
kubectl apply -f 00-namespace.yaml
kubectl apply -f 00b-storageclass.yaml
# create the real secret here (see 01-secrets.example.yaml header)
kubectl apply -f 02-configmap.yaml
kubectl apply -f 03-postgres.yaml      # wait: kubectl -n phishsentry rollout status statefulset/postgres
kubectl apply -f 04-redis.yaml
kubectl apply -f 05-inference.yaml     # wait for readiness (model load)
kubectl apply -f 06-rag.yaml           # wait for startupProbe (model load)
kubectl apply -f 07-app.yaml
kubectl apply -f 08-worker.yaml
kubectl apply -f 09-ingress.yaml       # ALB appears after ~2-3 min
kubectl apply -f 10-hpa.yaml
```

## After deploy
- `kubectl -n phishsentry get ingress phishsentry` → copy the ALB DNS name.
- In Cloudflare: CNAME-flatten the `phishsentry.app` apex to that ALB DNS name.
- Verify at `https://phishsentry.app`.

## Exposure model (preserved from the hardened compose)
Only the ALB (via the Ingress) is internet-facing. Every other component
(app, inference, rag, worker, postgres, redis) is a ClusterIP / headless
Service reachable **only inside the cluster** — the K8s equivalent of the
"nothing public but 443" posture. Observability (Prometheus/Grafana/Jaeger)
is intentionally not exposed here; add it as ClusterIP-only + SSH/port-forward
access, never through the public Ingress.

## Not yet included (deliberate next steps)
- **Observability manifests** (Prometheus/Grafana/Jaeger as ClusterIP).
- **NetworkPolicies** to enforce pod-to-pod least privilege (strong signal for a
  security project; requires NetworkPolicy enforcement enabled on the VPC CNI).
- **Dockerfile.rag** change to bake the ~2.4GB BGE models into the image
  (so `06-rag.yaml` needs no model PVC) — pending your current Dockerfile.rag.
