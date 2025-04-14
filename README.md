### Recommendations:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requrements.py
```

### Run code 
```
python main.py
```

### Run tests
```
PYTHONPATH=. pytest tests/test_strategies.py -v
```

### Strategies
**VWAP** stands for Valume-Weighted Average Price
It takes into account the Valume of shares which changed hands 
So the begger the volume the stronger the average value of that average price

**multi momentum** combines multiple metrics but with the different time period to make a choice
The idea is to combine benefits and increase the overall view
In our example we use RSI on 1 min and MACD on 15 min periods

**ATR with breakout** ATR stands for Average True Range
It calculates the volatility of market by subtracting values High Low and Close to find the beggest diviation and than sum it on some time period 
If its big the stop loss and take profit needs to further from the currend price
When the time goes and the new values comes we could move ower values with the trend to maximise the profit
We combine it with breakout of some price barier to determine the optimal entrance poin to the marken 
If its up breakout we want to go long if down - short

### Results
✅ Враховуй той факт що значення закриття це значення по тому за яку ціну був виконаний останній продаж чи покупка на ці значення орієнтуватись не можна

Час на який ми орієнтуємось при розробці стратегії - 1 хв 1 год 1 день

Перемістити ключі у нову локацію яку не закину на гітхаб

Написати кращу стратегію з ВВ і RSI

Підбір гіперпараметрві агрегації вікна 

Не втрачати конекшин

Якщо баланс почне падати зупинятись, значить нас вичислили і обхитрили

Запуск коду на сервері

Шортси Перехід на фючерси Перехід на інші криптовалюти
