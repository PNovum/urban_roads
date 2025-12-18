.PHONY: up down reset rebuild refresh train infer all logs wait

up:
	docker compose up -d

down:
	docker compose down

reset:
	docker compose down -v
	docker compose up -d

rebuild:
	docker compose build --no-cache

wait:
	docker compose up -d postgres mlflow
	docker compose exec -T postgres sh -lc 'until pg_isready -U $$PG_USER -d $$PG_DB; do sleep 1; done'
	docker compose exec -T mlflow sh -lc 'python -c "import urllib.request,time,sys; url=\"http://localhost:5000/\"; \
ok=False; \
exec(\"for _ in range(120):\\n try:\\n  urllib.request.urlopen(url, timeout=2); ok=True; break\\n except Exception:\\n  time.sleep(1)\"); \
sys.exit(0 if ok else 1)"'

refresh:
	docker compose run --rm pipeline python run.py --refresh

train:
	docker compose run --rm pipeline python run.py --train

infer:
	docker compose run --rm pipeline python run.py --infer

all:
	docker compose down -v
	docker compose up -d
	docker compose run --rm pipeline python run.py --all

logs:
	docker compose logs -f
