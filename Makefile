refresh:
	docker compose run --rm pipeline python run.py --refresh

infer:
	docker compose run --rm pipeline python run.py --infer

train:
	docker compose run --rm pipeline python run.py --train

all:
	docker compose run --rm pipeline python run.py --all

up:
	docker compose up -d --build

down:
	docker compose down -v
