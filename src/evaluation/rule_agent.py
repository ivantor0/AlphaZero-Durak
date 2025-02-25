# src/evaluation/rule_agent.py
import random
import numpy as np

class RuleAgent:
    """A rule-based agent for playing Durak."""
    def __init__(self):
        self._num_players = 2
        self._num_cards = 36
    
    def step(self, state):
        """Choose an action based on simple rules."""
        legal_actions = state.legal_actions()
        if not legal_actions:
            # If no legal actions, we should never get here in normal gameplay
            # But just in case, return a safe fallback
            print("WARNING: No legal actions available for rule agent!")
            return -1  # This will cause an error, but helps identify the issue
        
        # Get current player and phase
        current_player = state.current_player()
        phase = state._phase  # Access internal state
        
        # Extract useful information from state
        my_hand = state._hands[current_player]
        trump_suit = state._trump_suit
        table_cards = state._table_cards  # List of (atk_card, def_card_or_None)
        
        action = None
        
        if phase == 1:  # ATTACK
            action = self._attack(legal_actions, my_hand, trump_suit, table_cards)
        elif phase == 2:  # DEFENSE
            action = self._defend(legal_actions, my_hand, trump_suit, table_cards)
        elif phase == 3:  # ADDITIONAL
            action = self._additional_attack(legal_actions, my_hand, trump_suit, table_cards)
        elif phase == 4:  # PENDING_TAKE
            action = self._pending_take(legal_actions, my_hand, trump_suit, table_cards)
        else:
            # If we don't know what to do, select randomly
            action = random.choice(legal_actions)
        
        # Validate that we're returning a valid integer action
        if not isinstance(action, int) or action not in legal_actions:
            print(f"WARNING: Rule agent selected invalid action: {action}")
            # Fall back to a safe action
            action = legal_actions[0]
        
        return action
    
    def _attack(self, legal_actions, my_hand, trump_suit, table_cards):
        """Strategy for attacking."""
        # If we can finish the attack, prefer that
        if 37 in legal_actions:  # FINISH_ATTACK
            if len(table_cards) > 0:
                return 37
        
        # If this is the initial attack, play lowest non-trump if possible
        if len(table_cards) == 0:
            non_trump_cards = [card for card in my_hand if self._suit_of(card) != trump_suit]
            if non_trump_cards:
                lowest_non_trump = min(non_trump_cards, key=lambda x: self._rank_of(x))
                if lowest_non_trump in legal_actions:
                    return lowest_non_trump
            
            # If no non-trump, play lowest card
            lowest_card = min(my_hand, key=lambda x: (self._suit_of(x) == trump_suit, self._rank_of(x)))
            if lowest_card in legal_actions:
                return lowest_card
        
        # If we're continuing the attack, try to play cards of the same rank as already on table
        ranks_on_table = set(self._rank_of(ac) for (ac, dc) in table_cards)
        ranks_on_table.update(self._rank_of(dc) for (ac, dc) in table_cards if dc is not None)
        
        for rank in ranks_on_table:
            matching_cards = [card for card in my_hand if self._rank_of(card) == rank and card in legal_actions]
            if matching_cards:
                # Prefer non-trump
                non_trump_matches = [card for card in matching_cards if self._suit_of(card) != trump_suit]
                if non_trump_matches:
                    return random.choice(non_trump_matches)
                return random.choice(matching_cards)
        
        # If we can't play any matching cards, finish the attack
        if 37 in legal_actions:  # FINISH_ATTACK
            return 37
        
        # If we don't know what to do, select randomly from legal cards
        card_actions = [a for a in legal_actions if a < 36]
        if card_actions:
            return random.choice(card_actions)
        
        return random.choice(legal_actions)
    
    def _defend(self, legal_actions, my_hand, trump_suit, table_cards):
        """Strategy for defending."""
        # Check if we can finish the defense
        if 38 in legal_actions:  # FINISH_DEFENSE
            # If all cards are covered, finish defense
            all_covered = all(dc is not None for (ac, dc) in table_cards)
            if all_covered:
                return 38
        
        # Check if we should take cards
        if 36 in legal_actions:  # TAKE_CARDS
            uncovered_count = sum(1 for (ac, dc) in table_cards if dc is None)
            if uncovered_count > 2:  # If we need to defend against more than 2 cards, consider taking
                return 36
        
        # Try to defend with non-trump cards first
        for i, (ac, dc) in enumerate(table_cards):
            if dc is None:  # Uncovered attacking card
                # Find non-trump cards that can beat it
                non_trump_defenders = [
                    card for card in my_hand if card in legal_actions
                    and self._suit_of(card) == self._suit_of(ac)
                    and self._rank_of(card) > self._rank_of(ac)
                ]
                
                if non_trump_defenders:
                    # Use the lowest valid card
                    return min(non_trump_defenders, key=lambda x: self._rank_of(x))
                
                # If no non-trump defenders, look for valid trump cards
                trump_defenders = [
                    card for card in my_hand if card in legal_actions
                    and self._suit_of(card) == trump_suit
                ]
                
                if trump_defenders and self._suit_of(ac) != trump_suit:
                    # Use the lowest trump
                    return min(trump_defenders, key=lambda x: self._rank_of(x))
        
        # If we can't defend, take cards
        if 36 in legal_actions:  # TAKE_CARDS
            return 36
        
        # If we don't know what to do, select randomly
        return random.choice(legal_actions)
    
    def _additional_attack(self, legal_actions, my_hand, trump_suit, table_cards):
        """Strategy for additional attacking after all cards are covered."""
        # Same as regular attack, but we might be more aggressive
        return self._attack(legal_actions, my_hand, trump_suit, table_cards)
    
    def _pending_take(self, legal_actions, my_hand, trump_suit, table_cards):
        """Strategy for adding cards when defender has decided to take."""
        # Make sure FINISH_ATTACK is in legal actions before we rely on it
        finish_attack_action = 37  # FINISH_ATTACK constant
        
        try:
            # Get ranks currently on the table
            ranks_on_table = set(self._rank_of(ac) for (ac, dc) in table_cards)
            ranks_on_table.update(self._rank_of(dc) for (ac, dc) in table_cards if dc is not None)
            
            # Check if we have matching rank cards in our hand
            matching_cards = [card for card in my_hand if card in legal_actions and 
                             self._rank_of(card) in ranks_on_table]
            
            # If we have matching cards, play one
            if matching_cards:
                # Try to add lowest non-trump cards first
                non_trump_matches = [card for card in matching_cards if self._suit_of(card) != trump_suit]
                if non_trump_matches:
                    return min(non_trump_matches, key=lambda x: self._rank_of(x))
                
                # Or any matching card if that's all we have
                return min(matching_cards, key=lambda x: self._rank_of(x))
            
            # If no matching cards or other conditions apply, finish the attack
            if finish_attack_action in legal_actions:
                return finish_attack_action
        
        except Exception as e:
            print(f"Error in _pending_take: {e}")
            # If any error occurs, try to return a safe action
        
        # Final fallback - return the first legal action
        return legal_actions[0]
    
    def _suit_of(self, card):
        """Get the suit of a card (0-3)."""
        return card // 9
    
    def _rank_of(self, card):
        """Get the rank of a card (0-8)."""
        return card % 9