


#%%
class TeamTracker: 
    
    def __init__(self, initial_unseen_teams, initial_seen_teams):
        self.initial_unseen_teams = initial_unseen_teams.copy()
        self.initial_seen_teams = initial_seen_teams.copy()

        self.unseen_teams = initial_unseen_teams.copy()
        self.seen_teams = initial_seen_teams.copy()

    def reset(self):
        self.unseen_teams = self.initial_unseen_teams.copy()       
        self.seen_teams = self.initial_seen_teams.copy()
        #print('Team-tracker says: succesfully reset teams')
        #print('Teams now tracked: ')
        self.return_teams 

    def update_teams(self, seen_team):
        #print("Before update with team : ", seen_team)
        _a,_b = self.return_teams()
        self.unseen_teams.remove(seen_team)
        self.seen_teams.add(seen_team)
        #print("after update with seen_team = ", seen_team)
        _a, _b = self.return_teams()
    def return_teams(self):
        # if len(self.unseen_teams) == 0:
        #     print("unseen_teams = empty set ")
        # if self.seen_teams == 0:
        #     print("seen_teams = empty set ")

        # print("unseen_teams: ", self.unseen_teams)
        # print("seen_teams: ", self.seen_teams) 
        return self.unseen_teams, self.seen_teams
