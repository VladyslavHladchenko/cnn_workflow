import ipywidgets as widgets
import os 
from os.path import isdir, join
from utils import get_model_desc
class SelectMultipleInteract(widgets.HBox):

    def __init__(self, selected_list, dir='runs'):

        self.all_data_folders = [ join(dir, f) for f in os.listdir(dir) if isdir(join(dir, f))]
        self.options = list(map(get_model_desc, self.all_data_folders))

        self.selector = widgets.SelectMultiple(
            options=self.options,
            value=self.options,
            description='Models',
            disabled=False
        )
        self.selected_list=selected_list

        self._observed_function(self.selector)
        super().__init__(children=[self.selector])
        self._set_observes()

    def _set_observes(self):
        self.selector.observe(self._observed_function, names='value')

    def _observed_function(self, widg):
        self.selected_list.clear()
        for sl in self.selector.get_interact_value():
            idx = self.options.index(sl)
            self.selected_list.append(self.all_data_folders[idx])

