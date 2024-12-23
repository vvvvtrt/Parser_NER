import random


class Planning_Tours:
    def __init__(self, places=[], hobby=[], weather=[]):
        self.places = places
        self.hobby = hobby
        self.weather = weather

    def get_tour(self):
        weather_forecast = [i[1] for i in self.weather]
        max_places_per_day = max(2, min(5,
                                 len([place for place in self.places if place[1] == 'открытое']) // len(self.weather),
                                 len([place for place in self.places if place[1] == 'закрытое']) // len(self.weather)))

        unique_places = set()
        daily_recommendations = {}

        for day, weather in self.weather:
            if weather == 'хорошая':
                available_places = [place for place in self.places if
                                    place[1] == 'открытое' and place[0] not in unique_places]
            else:
                available_places = [place for place in self.places if
                                    place[1] == 'закрытое' and place[0] not in unique_places]


            if available_places:
                selected_places = random.sample(available_places, min(max_places_per_day, len(available_places)))
                daily_recommendations[f'{day}'] = selected_places
                unique_places.update(place[0] for place in selected_places)

        return daily_recommendations
