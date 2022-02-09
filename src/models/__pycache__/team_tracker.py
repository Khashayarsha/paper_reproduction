


#%%
class TeamTracker: 
    
    def __init__(self, initial_unseen_teams, initial_seen_teams):
        self.unseen_teams = initial_unseen_teams
        self.seen_teams =  initial_seen_teams

        self.initial_unseen_teams = initial_unseen_teams
        self.initial_seen_teams = initial_seen_teams

    def reset(self):
        self.unseen_teams = self.initial_unseen_teams       
        self.seen_teams = self.initial_seen_teams

    def update_seen_teams(self, seen_team):
        
        self.unseen_teams.remove(seen_team)
        self.seen_teams.add(seen_team)
    def return_teams(self):
        print("unseen_teams: ", self.unseen_teams)
        print("seen_teams: ", self.seen_teams) 
        return self.unseen_teams, self.seen_teams 