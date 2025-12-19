refresh:
	docker compose run --rm pipeline python run.py --refresh

train:
	docker compose run --rm pipeline python run.py --train

infer:
	docker compose run --rm pipeline python run.py --infer

all:
	docker compose run --rm pipeline python run.py --all

up:
	docker compose up -d --build

down:
	docker compose down -v
