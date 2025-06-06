.PHONY: all clean actions

include .env
export $(shell sed 's/=.*//' .env)

## FLAGS:
LOG_LEVEL = -vv
RASA_ENDPOINTS = --endpoints 
ENDPOINTS = endpoints.yml
RASA_CREDENTIALS = --credentials
CREDENTIALS = credentials.yml

# CLEAN PROJECT
clean:
	sudo rm -rf graph.html \
	       results/ \
	       models/* \
	       *.db* \
	       .*tensorboard* \
	       .rasa

install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

# RUN
api:
	rasa run \
		-m models/ \
		$(LOG_LEVEL) \
		$(RASA_ENDPOINTS) $(ENDPOINTS) \
		$(RASA_CREDENTIALS)	$(CREDENTIALS) \
		--enable-api \
		--cors '*' \
		--port 5005

shell:
	rasa shell \
		-m models/ \
		$(LOG_LEVEL) \
		$(RASA_ENDPOINTS) $(ENDPOINTS) \
		--port 5005

webchat:
	rasa run \
		-m models/ \
		$(LOG_LEVEL) \
		$(RASA_ENDPOINTS) $(ENDPOINTS) \
		$(RASA_CREDENTIALS) $(CREDENTIALS) \
		--cors '*' \
		--port 5005 

telegram:
	rasa run \
		-m models/ \
		$(LOG_LEVEL) \
		$(RASA_ENDPOINTS) $(ENDPOINTS) \
		$(RASA_CREDENTIALS)	credentials.yml \
		--port 5005 \
		--logging-config logging.yml

# ACTIONS
actions:
	rasa run actions \
		--actions actions \
		$(LOG_LEVEL)

# DEVELOPMENT
train-nlu:
	rasa train nlu \
		$(LOG_LEVEL)

train:
	rasa train \
		$(LOG_LEVEL)

test:
	rasa test \
		--out results/

test-nlu:
	rasa test nlu \
		--out results/results-nlu-test

test-core:
	rasa test core \
		--fail-on-prediction-errors \
		--out results/results-core-test

# UTIL
## RASALIT
install-rasalit:
	pip install -r rasalit-requirements.txt

rasalit:
	python -m rasalit nlu-cluster --port 8501

interactive:
	rasa interactive

validate:
	rasa data validate \
		$(LOG_LEVEL)

visualize:
		rasa visualize \
			--out ./graph.html

tensor:
		tensorboard \
			--logdir .

