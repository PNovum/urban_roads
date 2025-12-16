up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

refresh:
	docker compose run --rm pipeline python refresh.py --taxi taxi.xlsx --lt lt.xlsx --dist dist_cent.xlsx

psql:
	docker exec -it ur-postgres psql -U urmvp -d urmvp
